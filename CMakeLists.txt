cmake_minimum_required(VERSION 3.10)
project(yolov5_trt_api)

set(CMAKE_CXX_STANDARD 17)

# cuda
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)


# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# TensorRT
INCLUDE_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-8.6.1.6/include)
LINK_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-8.6.1.6/lib)


CUDA_ADD_LIBRARY(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

add_executable(yolov5_trt_api
        main.cpp yololayer.cu)

target_link_libraries(yolov5_trt_api myplugins)
target_link_libraries(yolov5_trt_api cudart)
target_link_libraries(yolov5_trt_api nvinfer)
target_link_libraries(yolov5_trt_api nvinfer_plugin)

TARGET_LINK_LIBRARIES(yolov5_trt_api ${OpenCV_LIBS})



