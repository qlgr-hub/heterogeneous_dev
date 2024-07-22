cmake_minimum_required(VERSION 3.25)

enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_COMPUTE_CAPABILITY 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_NVCC_FLAGS -G;-g)
# set(CUDA_NVCC_FLAGS -O3)

project(main)
add_executable(smemsquare checksmemsquare.cu)
set_target_properties(smemsquare PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(smemrectangle checksmemrectangle.cu)
set_target_properties(smemrectangle PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(reduce reduce.cu)
set_target_properties(reduce PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(transpose transpose.cu ../common/common.cu)
set_target_properties(transpose PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

# target_link_libraries(sumArray "-lcublas")
