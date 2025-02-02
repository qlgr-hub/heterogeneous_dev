cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(ch03)
enable_language(CUDA)
add_executable(reduce reduce.cu)
add_executable(coop3d coop3d.cu)
add_executable(showtile showtile.cu)