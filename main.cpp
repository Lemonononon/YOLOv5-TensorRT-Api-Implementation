#include <NvInfer.h>
#include <map>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstring>
#include <NvInferPlugin.h>

#include "logging.hpp"
#include "config.h"
#include "types.h"
#include "yololayer.hpp"

const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

using namespace nvinfer1;

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

ILayer* addFocus( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int input_h, int input_w ){

    ISliceLayer *s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{3, input_h / 2, input_w / 2}, Dims3{1, 2, 2});
    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    return addConvBNLeaky( network, weightMap, *cat->getOutput(0), 48, 3, 1, 1, "model.0.conv" );
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
ILayer* addBottleneckBlock( INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string name ) {

    int c_ = int(c2 * e);

    auto layer = addBottleneck( network, weightMap, input, c1, c2, shortcut, g, e, name+".0" );

    for (int i = 1; i < n; ++i) {
        layer = addBottleneck( network, weightMap, *layer->getOutput(0), c1, c2, shortcut, g, e, name+"."+std::to_string(i) );
    }
    return layer;
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
    auto focus0 = addFocus( network, weightMap, *data, kInputH, kInputW );
    printTensor("focus0", focus0->getOutput(0));

    auto conv1 = addConvBNLeaky( network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1,"model.1" );
    printTensor("conv1", conv1->getOutput(0));

    auto bottleneck2 = addBottleneckBlock( network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2" );
    printTensor("bottleneck2", bottleneck2->getOutput(0));

    auto conv3 = addConvBNLeaky( network, weightMap, *bottleneck2->getOutput(0), 192, 3, 2, 1, "model.3" );
    printTensor("conv3", conv3->getOutput(0));

    auto bottleneck_csp4 = addBottleneckCSP( network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4" );
    printTensor("bottleneck_csp4", bottleneck_csp4->getOutput(0));

    auto conv5 = addConvBNLeaky( network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5" );
    printTensor("conv5", conv5->getOutput(0));

    auto bottleneck_csp6 = addBottleneckCSP( network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6" );
    printTensor("bottleneck_csp6", bottleneck_csp6->getOutput(0));

    auto conv7 = addConvBNLeaky( network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    printTensor("conv7", conv7->getOutput(0));

    auto spp8 = addSPP( network, weightMap, *conv7->getOutput(0), 768, 768, {5, 9, 13}, "model.8" );
    printTensor("spp8", spp8->getOutput(0));

    auto bottleneck_csp9 = addBottleneckCSP( network, weightMap, *spp8->getOutput(0), 768, 768, 4, true, 1, 0.5, "model.9" );
    printTensor("bottleneck_csp9", bottleneck_csp9->getOutput(0));

    // head
    auto bottleneck_csp10 = addBottleneckCSP( network, weightMap, *spp8->getOutput(0), 768, 768, 2, true, 1, 0.5, "model.10" );
    printTensor("bottleneck_csp10", bottleneck_csp10->getOutput(0));

    auto conv11 = network->addConvolutionNd( *bottleneck_csp10->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.11.weight"], weightMap["model.11.bias"]);
    conv11->setName("conv.11");
    conv11->setStrideNd(DimsHW{1, 1});
    conv11->setPaddingNd(DimsHW{0, 0});
    printTensor("conv11", conv11->getOutput(0));

    float scale[] = {1.0, 2.0, 2.0};
    //upsample12 scale_factor = 2
    auto upsample12 = network->addResize(*bottleneck_csp10->getOutput(0));
    upsample12->setResizeMode(ResizeMode::kNEAREST);
    upsample12->setScales(scale, 3);
    printTensor("upsample12", upsample12->getOutput(0));

    ITensor* inputTensorsCat13[] = {upsample12->getOutput(0), conv5->getOutput(0)};
    auto cat13 = network->addConcatenation(inputTensorsCat13, 2);
    printTensor("cat13", cat13->getOutput(0));

    auto conv14 = addConvBNLeaky( network, weightMap, *cat13->getOutput(0), 384, 1, 1, 1, "model.14" );
    printTensor("conv14", conv14->getOutput(0));

    auto bottleneck_csp15 = addBottleneckCSP( network, weightMap, *conv14->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.15" );
    printTensor("bottleneck_csp15", bottleneck_csp15->getOutput(0));

    auto conv16 = network->addConvolutionNd( *bottleneck_csp15->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.16.weight"], weightMap["model.16.bias"]);
    printTensor("conv16", conv16->getOutput(0));

    // upsample 17
    auto upsample17 = network->addResize( *bottleneck_csp15->getOutput(0) );
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setScales(scale, 3);
    printTensor("upsample17", upsample17->getOutput(0));


    ITensor* inputTensorsCat18[] = {upsample17->getOutput(0), conv3->getOutput(0)};
    auto cat18 = network->addConcatenation(inputTensorsCat18, 2);
    printTensor("cat18", cat18->getOutput(0));

    auto conv19 = addConvBNLeaky(network, weightMap, *cat18->getOutput(0), 192, 1, 1, 1, "model.19");
    printTensor("conv19", conv19->getOutput(0));

    auto bottleneck_csp20 = addBottleneckCSP( network, weightMap, *conv19->getOutput(0), 192, 192, 2, false, 1, 0.5, "model.20" );
    printTensor("bottleneck_csp20", bottleneck_csp20->getOutput(0));

    auto conv21 = network->addConvolutionNd( *bottleneck_csp20->getOutput(0), 24, DimsHW{1, 1}, weightMap["model.21.weight"], weightMap["model.21.bias"]);
    printTensor("conv21", conv21->getOutput(0));

    // Detect plugin
    auto yolo = addYololayer( network, weightMap, "model.22", {conv11, conv16, conv21} );

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    auto engine = builder->buildSerializedNetwork(*network, *config);

    return engine;

}

static Logger gLogger;

int main(){

    initLibNvInferPlugins(&gLogger, "");

    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    auto engine = build_engine( 1, builder, config, DataType::kFLOAT, "/home/xiaoying/code/yolov5_trt_api/weights/yolov5m.wts");





}