cmake_minimum_required(VERSION 3.25)

enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(sumArrays)
add_executable(sumArraysOnHost sumArraysOnHost.c)
add_executable(checkDimension checkDimension.cu)
add_executable(defineGridBlock defineGridBlock.cu)
add_executable(sumArraysOnGPU-small-case sumArraysOnGPU-small-case.cu)
add_executable(sumArraysOnGPU-timer sumArraysOnGPU-timer.cu)
add_executable(checkThreadIndex checkThreadIndex.cu)
add_executable(sumMatrixOnGPU-2D-grid-2D-block sumMatrixOnGPU-2D-grid-2D-block.cu)
add_executable(sumMatrixOnGPU-1D-grid-1D-block sumMatrixOnGPU-1D-grid-1D-block.cu)
add_executable(sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU-2D-grid-1D-block.cu)
add_executable(checkDeviceInfor checkDeviceInfor.cu)

# target_link_libraries(sumArray "-lcublas")
