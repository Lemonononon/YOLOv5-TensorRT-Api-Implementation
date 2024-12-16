#include <NvInfer.h>
#include <map>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstring>
#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>

#include "logging.hpp"
#include "config.h"
#include "types.h"
#include "plugin/yololayer.hpp"

const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

using namespace nvinfer1;
static Logger gLogger;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
static std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

ILayer* addBN( INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string name, float eps) {

    float* gamma = (float*)weightMap[name + ".weight"].values;
    float* beta = (float*)weightMap[name + ".bias"].values;
    float* mean = (float*)weightMap[name + ".running_mean"].values;
    float* var = (float*)weightMap[name + ".running_var"].values;
    int len = weightMap[name + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[name + ".scale"] = scale;
    weightMap[name + ".shift"] = shift;
    weightMap[name + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;

}


ILayer* addConvBNLeaky( INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, int g, std::string name ) {

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto conv = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[name + ".conv.weight"], emptywts);
    conv->setName((name + ".conv").c_str());
    conv->setStrideNd(DimsHW{s, s});
    conv->setNbGroups(g);

    int p = k / 2;
    conv->setPaddingNd(DimsHW{p, p});

    auto bn = addBN( network, weightMap, *conv->getOutput(0), name+".bn", 0.0001 );

    // LeakyRelu
    auto leaky = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky->setAlpha(0.1);
    leaky->setName((name + ".leaky").c_str());

    return leaky;

}

ILayer* addFocus( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int input_h, int input_w, int output_c, int kernel_size, int stride, int g ){

    ISliceLayer *s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    return addConvBNLeaky( network, weightMap, *cat->getOutput(0), output_c, kernel_size, stride, g, "model.0.conv" );
}

ILayer* addBottleneck( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string name ) {

    // c_ = int(c2 * e)  # hidden channels
    // self.cv1 = Conv(c1, c_, 1, 1)
    // self.cv2 = Conv(c_, c2, 3, 1, g=g)
    // self.add = shortcut and c1 == c2

    int c_ = c2 * e;

    auto cv1 = addConvBNLeaky( network, weightMap, input, c_, 1, 1, 1, name + ".cv1" );
    auto cv2 = addConvBNLeaky( network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, name + ".cv2" );

    if ( shortcut && (c1 == c2)) {
        return network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
    }

    return cv2;
}
ILayer* addBottleneckBlock( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string name, bool need_suffix = true ) {

    if (need_suffix){
        auto layer = addBottleneck( network, weightMap, input, c1, c2, shortcut, g, e, name+".0" );

        for (int i = 1; i < n; ++i) {
            layer = addBottleneck( network, weightMap, *layer->getOutput(0), c1, c2, shortcut, g, e, name+"."+std::to_string(i) );
        }

        return layer;

    }else{
        auto layer = addBottleneck( network, weightMap, input, c1, c2, shortcut, g, e, name );

        for (int i = 1; i < n; ++i) {
            layer = addBottleneck( network, weightMap, *layer->getOutput(0), c1, c2, shortcut, g, e, name );
        }
        return layer;
    }

}

ILayer* addBottleneckCSP( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string name ) {

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // y1 = self.cv3(self.m(self.cv1(x)))
    // y2 = self.cv2(x)
    // return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    int c_ = c2 * e;

    auto cv1 = addConvBNLeaky( network, weightMap, input, c_, 1, 1, g, name + ".cv1" );
    auto cv2 = network->addConvolutionNd( input, c_, DimsHW{1, 1}, weightMap[name+".cv2.weight"], emptywts );
    cv2->setName((name + ".cv2").c_str());
    cv2->setStrideNd(DimsHW{1, 1});

   //  bottleneck
   auto bottleneck = addBottleneckBlock( network, weightMap, *cv1->getOutput(0), c_, c_, n, shortcut, g, 1.0, name+".m" );

   auto cv3 = network->addConvolutionNd( *bottleneck->getOutput(0), c_, DimsHW{1, 1}, weightMap[name+".cv3.weight"], emptywts);
   cv3->setName((name + ".cv3").c_str());
   cv3->setStrideNd(DimsHW{1, 1});

   ITensor *inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};

   auto cat = network->addConcatenation( inputTensors, 2);

   auto bn = addBN( network, weightMap, *cat->getOutput(0), name+".bn", 0.0001 );

   auto leaky = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
   leaky->setAlpha(0.1);

   auto cv4 = addConvBNLeaky( network, weightMap, *leaky->getOutput(0), c2, 1, 1, g, name + ".cv4");

   return cv4;
}

ILayer* addSPP( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, std::vector<int> k, std::string name ) {

    int c_ = c1 / 2;

    auto cv1 = addConvBNLeaky( network, weightMap, input, c_, 1, 1, 1, name + ".cv1");

    auto pool0 = network->addPoolingNd( *cv1->getOutput(0), PoolingType::kMAX, DimsHW{k[0], k[0]});
    pool0->setPaddingNd(DimsHW{k[0] / 2, k[0] / 2});
    pool0->setStrideNd(DimsHW{1, 1});

    auto pool1 = network->addPoolingNd( *cv1->getOutput(0), PoolingType::kMAX, DimsHW{k[1], k[1]});
    pool1->setPaddingNd(DimsHW{k[1] / 2, k[1] / 2});
    pool1->setStrideNd(DimsHW{1, 1});

    auto pool2 = network->addPoolingNd( *cv1->getOutput(0), PoolingType::kMAX, DimsHW{k[2], k[2]});
    pool2->setPaddingNd(DimsHW{k[2] / 2, k[2] / 2});
    pool2->setStrideNd(DimsHW{1, 1});

    ITensor* inputTensors[] = {cv1->getOutput(0), pool0->getOutput(0), pool1->getOutput(0), pool2->getOutput(0)};
    auto cat = network->addConcatenation( inputTensors, 4);

    return addConvBNLeaky( network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, name + ".cv2");
}

static std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = kNumAnchor * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

ILayer* addYololayer( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, std::string name, std::vector<IConvolutionLayer*> dets ){

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, name);

    PluginField plugin_fields[2];
    int netinfo[4] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;

    // stride: 8 16 32
    int scale = 8;

    std::vector<YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        YoloKernel kernel;
        kernel.width = kInputW / scale;
        kernel.height = kInputH / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;

}

void printTensor(const char* name, ITensor* tensor) {
    auto d = tensor->getDimensions();
    std::cout << name << ": ";
    for (int i = 0; i < d.nbDims; ++i) {
        std::cout << d.d[i] << " ";
    }
    std::cout <<std::endl;
}

IHostMemory* build_engine( unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path ){

    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, kInputH, kInputW}
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });

    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // bakcbone
    auto focus0 = addFocus( network, weightMap, *data, kInputH, kInputW, 48, 3, 1, 1 );
    // printTensor("focus0", focus0->getOutput(0));

    auto conv1 = addConvBNLeaky( network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1,"model.1" );
    // printTensor("conv1", conv1->getOutput(0));
    // conv1->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv1->getOutput(0));

    auto bottleneck2 = addBottleneckBlock( network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2" );
    // printTensor("bottleneck2", bottleneck2->getOutput(0));
    // bottleneck2->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck2->getOutput(0));

    auto conv3 = addConvBNLeaky( network, weightMap, *bottleneck2->getOutput(0), 192, 3, 2, 1, "model.3" );
    // printTensor("conv3", conv3->getOutput(0));

    auto bottleneck_csp4 = addBottleneckCSP( network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4" );
    // printTensor("bottleneck_csp4", bottleneck_csp4->getOutput(0));

    auto conv5 = addConvBNLeaky( network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5" );
    // printTensor("conv5", conv5->getOutput(0));
    // conv5->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv5->getOutput(0));


    auto bottleneck_csp6 = addBottleneckCSP( network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6" );
    // printTensor("bottleneck_csp6", bottleneck_csp6->getOutput(0));

    auto conv7 = addConvBNLeaky( network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    // printTensor("conv7", conv7->getOutput(0));

    auto spp8 = addSPP( network, weightMap, *conv7->getOutput(0), 768, 768, {5, 9, 13}, "model.8" );
    // printTensor("spp8", spp8->getOutput(0));

    auto bottleneck_csp9 = addBottleneckCSP( network, weightMap, *spp8->getOutput(0), 768, 768, 4, true, 1, 0.5, "model.9" );
    // printTensor("bottleneck_csp9", bottleneck_csp9->getOutput(0));

    // // head  wrong
    auto bottleneck_csp10 = addBottleneckCSP( network, weightMap, *bottleneck_csp9->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.10" );
    // printTensor("bottleneck_csp10", bottleneck_csp10->getOutput(0));
    // bottleneck_csp10->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck_csp10->getOutput(0));

    auto conv11 = network->addConvolutionNd( *bottleneck_csp10->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.11.weight"], weightMap["model.11.bias"]);
    conv11->setName("conv.11");
    conv11->setStrideNd(DimsHW{1, 1});
    conv11->setPaddingNd(DimsHW{0, 0});
    // printTensor("conv11", conv11->getOutput(0));

    float scale[] = {1.0, 2.0, 2.0};
    //upsample12 scale_factor = 2
    auto upsample12 = network->addResize(*bottleneck_csp10->getOutput(0));
    upsample12->setResizeMode(ResizeMode::kNEAREST);
    upsample12->setScales(scale, 3);
    // printTensor("upsample12", upsample12->getOutput(0));

    ITensor* inputTensorsCat13[] = {upsample12->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat13 = network->addConcatenation(inputTensorsCat13, 2);
    // printTensor("cat13", cat13->getOutput(0));
    // cat13->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*cat13->getOutput(0));

    //
    //
    //
    auto conv14 = addConvBNLeaky( network, weightMap, *cat13->getOutput(0), 384, 1, 1, 1, "model.14" );
    // printTensor("conv14", conv14->getOutput(0));
    // conv14->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv14->getOutput(0));


    auto bottleneck_csp15 = addBottleneckCSP( network, weightMap, *conv14->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.15" );
    // printTensor("bottleneck_csp15", bottleneck_csp15->getOutput(0));

    auto conv16 = network->addConvolutionNd( *bottleneck_csp15->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.16.weight"], weightMap["model.16.bias"]);
    // printTensor("conv16", conv16->getOutput(0));

    // conv16->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv16->getOutput(0));

    // upsample 17
    auto upsample17 = network->addResize( *bottleneck_csp15->getOutput(0) );
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setScales(scale, 3);
    // printTensor("upsample17", upsample17->getOutput(0));


    ITensor* inputTensorsCat18[] = {upsample17->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat18 = network->addConcatenation(inputTensorsCat18, 2);
    // printTensor("cat18", cat18->getOutput(0));
    // cat18->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*cat18->getOutput(0));

    auto conv19 = addConvBNLeaky(network, weightMap, *cat18->getOutput(0), 192, 1, 1, 1, "model.19");
    // printTensor("conv19", conv19->getOutput(0));
    // conv19->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv19->getOutput(0));

    auto bottleneck_csp20 = addBottleneckCSP( network, weightMap, *conv19->getOutput(0), 192, 192, 2, false, 1, 0.5, "model.20" );
    // printTensor("bottleneck_csp20", bottleneck_csp20->getOutput(0));
    // bottleneck_csp20->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck_csp20->getOutput(0));

    auto conv21 = network->addConvolutionNd( *bottleneck_csp20->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.21.weight"], weightMap["model.21.bias"]);
    // printTensor("conv21", conv21->getOutput(0));

    // // Detect plugin
    auto yolo = addYololayer( network, weightMap, "model.22", {conv21, conv16, conv11} );

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    config->setFlag(BuilderFlag::kFP16);
    auto engine = builder->buildSerializedNetwork(*network, *config);

    return engine;

}

IHostMemory* build_engine_s( unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path ){

    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, kInputH, kInputW}
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });

    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // bakcbone
    auto focus0 = addFocus( network, weightMap, *data, kInputH, kInputW, 32, 3 , 1, 1 );
    focus0->setName("focus0");
    // printTensor("focus0", focus0->getOutput(0));

    // focus0->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*focus0->getOutput(0));
    //
    // auto shape = focus0->getOutput(0)->getDimensions();
    //
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    //
    // std::cout << std::endl;




    auto conv1 = addConvBNLeaky( network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1,"model.1" );
    conv1->setName("conv1");
    // printTensor("conv1", conv1->getOutput(0));
    // conv1->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv1->getOutput(0));

    auto bottleneck2 = addBottleneckBlock( network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2",
                                           false );
    bottleneck2->setName("bottleneck2");
    // printTensor("bottleneck2", bottleneck2->getOutput(0));
    // bottleneck2->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck2->getOutput(0));

    auto conv3 = addConvBNLeaky( network, weightMap, *bottleneck2->getOutput(0), 128, 3, 2, 1, "model.3" );
    conv3->setName("conv3");
    // printTensor("conv3", conv3->getOutput(0));

    // conv3->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv3->getOutput(0));
    //
    // auto shape = conv3->getOutput(0)->getDimensions();
    //
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    //
    // std::cout << std::endl;


    auto bottleneck_csp4 = addBottleneckCSP( network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4" );
    bottleneck_csp4->setName("bottleneck_csp4");
    // printTensor("bottleneck_csp4", bottleneck_csp4->getOutput(0));

    auto conv5 = addConvBNLeaky( network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5" );
    conv5->setName("conv5");
    // // printTensor("conv5", conv5->getOutput(0));
    // conv5->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv5->getOutput(0));
    //
    // auto shape = conv5->getOutput(0)->getDimensions();
    //
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    //
    // std::cout << std::endl;


    auto bottleneck_csp6 = addBottleneckCSP( network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6" );
    bottleneck_csp6->setName("bottleneck_csp6");
    // printTensor("bottleneck_csp6", bottleneck_csp6->getOutput(0));

    auto conv7 = addConvBNLeaky( network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    conv7->setName("conv7");
    // printTensor("conv7", conv7->getOutput(0));

    auto spp8 = addSPP( network, weightMap, *conv7->getOutput(0), 512, 512, {5, 9, 13}, "model.8" );
    spp8->setName("spp8");
    // printTensor("spp8", spp8->getOutput(0));

    auto bottleneck_csp9 = addBottleneckCSP( network, weightMap, *spp8->getOutput(0), 512, 512, 2, true, 1, 0.5, "model.9" );
    bottleneck_csp9->setName("bottleneck_csp9");
    // printTensor("bottleneck_csp9", bottleneck_csp9->getOutput(0));

    // // head  wrong
    auto bottleneck_csp10 = addBottleneckCSP( network, weightMap, *bottleneck_csp9->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.10" );
    bottleneck_csp10->setName("bottleneck_csp10");
    // printTensor("bottleneck_csp10", bottleneck_csp10->getOutput(0));
    // bottleneck_csp10->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck_csp10->getOutput(0));

    auto conv11 = network->addConvolutionNd( *bottleneck_csp10->getOutput(0), 21, DimsHW{1, 1}, weightMap["model.11.weight"], weightMap["model.11.bias"]);
    conv11->setName("conv.11");
    conv11->setStrideNd(DimsHW{1, 1});
    conv11->setPaddingNd(DimsHW{0, 0});
    // printTensor("conv11", conv11->getOutput(0));

    float scale[] = {1.0, 2.0, 2.0};
    //upsample12 scale_factor = 2
    auto upsample12 = network->addResize(*bottleneck_csp10->getOutput(0));
    upsample12->setName("upsample12");
    upsample12->setResizeMode(ResizeMode::kNEAREST);
    upsample12->setScales(scale, 3);
    // printTensor("upsample12", upsample12->getOutput(0));

    ITensor* inputTensorsCat13[] = {upsample12->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat13 = network->addConcatenation(inputTensorsCat13, 2);
    cat13->setName("cat13");
    // printTensor("cat13", cat13->getOutput(0));
    // cat13->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*cat13->getOutput(0));

    //
    //
    //
    auto conv14 = addConvBNLeaky( network, weightMap, *cat13->getOutput(0), 256, 1, 1, 1, "model.14" );
    conv14->setName("conv14");
    // printTensor("conv14", conv14->getOutput(0));
    // conv14->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv14->getOutput(0));


    auto bottleneck_csp15 = addBottleneckCSP( network, weightMap, *conv14->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.15" );
    bottleneck_csp15->setName("bottleneck_csp15");
    // printTensor("bottleneck_csp15", bottleneck_csp15->getOutput(0));

    auto conv16 = network->addConvolutionNd( *bottleneck_csp15->getOutput(0), 21, DimsHW{1, 1}, weightMap["model.16.weight"], weightMap["model.16.bias"]);
    conv16->setName("conv16");
    // printTensor("conv16", conv16->getOutput(0));

    // conv16->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv16->getOutput(0));

    // upsample 17
    auto upsample17 = network->addResize( *bottleneck_csp15->getOutput(0) );
    upsample17->setName("upsample17");
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setScales(scale, 3);
    // printTensor("upsample17", upsample17->getOutput(0));


    ITensor* inputTensorsCat18[] = {upsample17->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat18 = network->addConcatenation(inputTensorsCat18, 2);
    cat18->setName("cat18");
    // printTensor("cat18", cat18->getOutput(0));
    // cat18->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*cat18->getOutput(0));

    auto conv19 = addConvBNLeaky(network, weightMap, *cat18->getOutput(0), 128, 1, 1, 1, "model.19");
    conv19->setName("conv19");
    // printTensor("conv19", conv19->getOutput(0));
    // conv19->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv19->getOutput(0));

    auto bottleneck_csp20 = addBottleneckCSP( network, weightMap, *conv19->getOutput(0), 128, 128, 1, false, 1, 0.5, "model.20" );
    bottleneck_csp20->setName("bottleneck_csp20");
    // printTensor("bottleneck_csp20", bottleneck_csp20->getOutput(0));
    // bottleneck_csp20->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*bottleneck_csp20->getOutput(0));

    auto conv21 = network->addConvolutionNd( *bottleneck_csp20->getOutput(0), 21, DimsHW{1, 1}, weightMap["model.21.weight"], weightMap["model.21.bias"]);
    conv21->setName("conv21");
    // // printTensor("conv21", conv21->getOutput(0));
    // conv21->getOutput(0)->setName(kOutputTensorName);
    // network->markOutput(*conv21->getOutput(0));
    //
    // auto shape = conv21->getOutput(0)->getDimensions();
    //
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    // std::cout << std::endl;


    // // Detect plugin
    auto yolo = addYololayer( network, weightMap, "model.22", {conv21, conv16, conv11} );
    yolo->setName("yolo");

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    config->setFlag(BuilderFlag::kFP16);
    auto engine = builder->buildSerializedNetwork(*network, *config);

    return engine;
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);

    std::vector<Detection> m;

    for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; ++i) {
        if ( output[ 1 + det_size * i + 4 ] <= conf_thresh ) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        m.push_back(det);
    }

    // 两种类别在一起做nms，而不是分别做一遍
    std::sort(m.begin(), m.end(), cmp);

    for (int i = 0; i < m.size(); ++i) {
        auto& item = m[i];
        res.push_back(item);
        for (int j = i + 1; j < m.size(); ++j) {
            if (iou(item.bbox, m[j].bbox) > nms_thresh) {
                m.erase(m.begin() + j);
                --j;
            }
        }
    }
}

void nms_platelet(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);

    std::vector<Detection> m;

    for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; ++i) {
        if ( output[ 1 + det_size * i + 4 ] <= conf_thresh ) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        m.push_back(det);
    }

    // 两种类别在一起做nms，而不是分别做一遍
    std::sort(m.begin(), m.end(), cmp);

    for (int i = 0; i < m.size(); ++i) {
        auto& item = m[i];
        res.push_back(item);
        for (int j = i + 1; j < m.size(); ++j) {
            if (iou(item.bbox, m[j].bbox) > nms_thresh) {
                m.erase(m.begin() + j);
                --j;
            }
        }
    }
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

void draw_bbox( cv::Mat& img, std::vector<Detection>& res) {

    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(img, res[j].bbox);

        cv::Scalar color;
        if ( res[j].class_id == 0 ){
            // green
            color = cv::Scalar(0,255,0);
        }else if ( res[j].class_id == 1){
            // blue
            color = cv::Scalar(255,0,0);
        }else{
            color = cv::Scalar(0,0,255);
        }
        cv::rectangle(img, r, color, 2);

        // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int)res[j].class_id) + " " + std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch) {
    for (size_t i = 0; i < img_batch.size(); i++) {
        auto& res = res_batch[i];
        cv::Mat img = img_batch[i];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id) + " " + std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }
}


int main(){

    std::string engine_path = "../weights/yolov5s_platelet.engine";
    bool need_build = false;
    need_build = true;

    initLibNvInferPlugins(&gLogger, "");

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    if (need_build){
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        auto engine_data = build_engine_s( 4, builder, config, DataType::kFLOAT, "/home/xiaoying/code/yolov5_trt_api/weights/yolov5s_platelet.wts");
        // auto engine_data = build_engine( 4, builder, config, DataType::kFLOAT, "/home/xiaoying/code/yolov5_trt_api/weights/yolov5m.wts");

        std::ofstream ofs(engine_path, std::ios::binary);
        if (!ofs){
            std::cout << "could not open " << engine_path << std::endl;
            return -1;
        }

        ofs.write((const char*)engine_data->data(), engine_data->size());

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engine_data->data(), engine_data->size());
        context = engine->createExecutionContext();

    }else {

        deserialize_engine(engine_path, &runtime, &engine, &context);

    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int batchSizeDetection = 1;
    int inputHeightDetection = 640;
    int inputWidthDetection = 640;

    // 输入输出绑定
    void* buffers[2]{};

    auto inputIndex = engine->getBindingIndex(kInputTensorName);
    auto outputIndex = engine->getBindingIndex(kOutputTensorName);

    // std::cout << "inputIndex: " << inputIndex << std::endl;
    // std::cout << "outputIndex: " << outputIndex << std::endl;

    int OUTPUT_SIZE = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    // OUTPUT_SIZE = 1152*40*40;

    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSizeDetection * 3 * inputHeightDetection * inputWidthDetection * sizeof(float))); //对gpu进行显存分配
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSizeDetection * OUTPUT_SIZE * sizeof(float)));

    // red
    // float CONF_THRESH = 0.35;
    // float NMS_THRESH = 0.2;

    // platelet
    float CONF_THRESH = 0.3;
    float NMS_THRESH = 0.3;


    std::vector<float> data;
    data.resize(batchSizeDetection * 3 * inputHeightDetection * inputWidthDetection);

    auto origin_img = cv::imread("/home/xiaoying/code/zhangqian/red_det/yolov5-master_rbc_fang/ceshi_plot/rt/132_057_93623_142857_0.jpg");
    cv::Mat img;

    cv::resize(origin_img, img, cv::Size(inputWidthDetection, inputHeightDetection));

    int index = 0;
    for (int row = 0; row < inputHeightDetection; ++row) {
        uchar *uc_pixel = img.data + row * img.step;
        for (int col = 0; col < inputWidthDetection; ++col) {
            data[index] = (float) uc_pixel[2] / 255.0;
            data[index + inputHeightDetection * inputWidthDetection] = (float) uc_pixel[1] / 255.0;
            data[index + 2 * inputHeightDetection * inputWidthDetection] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++index;
        }
    }

    std::vector<float> prob;
    prob.resize(1*OUTPUT_SIZE);

    cudaMemcpyAsync( buffers[inputIndex], data.data(), 1 * 3 * inputHeightDetection * inputWidthDetection * sizeof(float), cudaMemcpyHostToDevice, stream );
    // context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, inputHeightDetection, inputWidthDetection));
    // context->enqueueV2(buffers, stream, nullptr);
    context->enqueue(1, buffers, stream, nullptr);

    auto shape = engine->getTensorShape(kOutputTensorName);

    // std::cout << std::endl;
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    // std::cout << std::endl;

    CUDA_CHECK(cudaMemcpyAsync( prob.data(), buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream ));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    auto begin = std::chrono::system_clock::now();
    cv::resize(origin_img, img, cv::Size(inputWidthDetection, inputHeightDetection));

    index = 0;
    for (int row = 0; row < inputHeightDetection; ++row) {
        uchar *uc_pixel = img.data + row * img.step;
        for (int col = 0; col < inputWidthDetection; ++col) {
            data[index] = (float) uc_pixel[2] / 255.0;
            data[index + inputHeightDetection * inputWidthDetection] = (float) uc_pixel[1] / 255.0;
            data[index + 2 * inputHeightDetection * inputWidthDetection] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++index;
        }
    }

    cudaMemcpyAsync( buffers[inputIndex], data.data(), 1 * 3 * inputHeightDetection * inputWidthDetection * sizeof(float), cudaMemcpyHostToDevice, stream );
    // context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, inputHeightDetection, inputWidthDetection));
    // context->enqueueV2(buffers, stream, nullptr);
    context->enqueue(1, buffers, stream, nullptr);

    // auto shape = engine->getTensorShape(kOutputTensorName);
    //
    // for (int i = 0; i < shape.nbDims; ++i) {
    //     std::cout << shape.d[i] << " ";
    // }
    //
    // std::cout << std::endl;

    CUDA_CHECK(cudaMemcpyAsync( prob.data(), buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //
    //
    // // for (int i = prob.size()-21; i < prob.size(); ++i) {
    // //     std::cout << prob[i] << " ";
    // // }
    //
    // // nms
    std::vector<std::vector<Detection>> batch_res(1);
    for (int b = 0; b < 1; b++) {
        auto& res = batch_res[b];
        nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "inference time: " << duration.count() << " ms" << std::endl;

    std::cout << "num of boxes: " << batch_res[0].size() << std::endl;

    draw_bbox(origin_img, batch_res[0]);

    cv::imwrite("result.jpg", origin_img);

}