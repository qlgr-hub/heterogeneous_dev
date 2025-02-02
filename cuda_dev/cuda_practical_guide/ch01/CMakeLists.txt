cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(ch01)
enable_language(CUDA)
add_executable(ch01 main.cu)
target_include_directories(ch01 PUBLIC ../common /usr/local/cuda/include)
target_link_libraries(ch01 PUBLIC omp5)