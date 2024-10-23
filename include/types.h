//
// Created by xiaoying on 24-10-22.
//

#ifndef YOLOV5_TRT_API_TYPES_HPP
#define YOLOV5_TRT_API_TYPES_HPP

#include "config.h"

struct YoloKernel {
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};


#endif //YOLOV5_TRT_API_TYPES_HPP
