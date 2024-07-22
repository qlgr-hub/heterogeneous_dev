cmake_minimum_required(VERSION 3.25)

project(cuda_hello)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(hellocuda helloGPU.cu)
