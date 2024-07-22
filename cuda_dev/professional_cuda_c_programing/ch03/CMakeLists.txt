cmake_minimum_required(VERSION 3.25)

enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CUDA_NVCC_FLAGS -G;-g)
set(CUDA_NVCC_FLAGS -O3)

project(simpleDivergence)
add_executable(simpleDivergence simpleDivergence.cu)
add_executable(simpleDeviceQuery simpleDeviceQuery.cu)
add_executable(reduceInteger reduceInteger.cu)

# target_link_libraries(sumArray "-lcublas")
