cmake_minimum_required(VERSION 3.18)
project(cuda_vision_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3050 (Ampere SM_86)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(vision_pipeline src/main.cpp src/vision_kernels.cu)
target_link_libraries(vision_pipeline ${OpenCV_LIBS} Threads::Threads)
