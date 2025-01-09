cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)
set(CUDACXX /usr/local/cuda/bin/nvcc)
project(yolov5_trt_api LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# cuda
#FIND_PACKAGE(CUDA REQUIRED)
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)



# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# TensorRT
#INCLUDE_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-8.6.1.6/include)
#LINK_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-8.6.1.6/lib)
INCLUDE_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-10.6.0.26/include)
LINK_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-10.6.0.26/lib)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

ADD_LIBRARY(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

add_executable(yolov5_trt_api
        main.cpp plugin/yololayer.cu)

target_link_libraries(yolov5_trt_api myplugins)
target_link_libraries(yolov5_trt_api cudart)
target_link_libraries(yolov5_trt_api nvinfer)
target_link_libraries(yolov5_trt_api nvinfer_plugin)

TARGET_LINK_LIBRARIES(yolov5_trt_api ${OpenCV_LIBS})



