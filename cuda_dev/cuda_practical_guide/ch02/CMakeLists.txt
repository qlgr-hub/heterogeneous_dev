cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(tiling)
enable_language(CUDA)
add_executable(tiling tiling.cu)
target_include_directories(tiling PUBLIC ../common /usr/local/cuda/include)

add_executable(reduce reduce.cu)
add_executable(sharedmemory sharedmemory.cu)
add_executable(matmul matmul.cu)
target_link_libraries(matmul cublas)