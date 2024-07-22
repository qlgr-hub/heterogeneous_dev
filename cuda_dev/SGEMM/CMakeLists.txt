cmake_minimum_required(VERSION 3.25)

enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_COMPUTE_CAPABILITY 89)
# set(CUDA_NVCC_FLAGS -G;-g)
# set(CUDA_NVCC_FLAGS -O3)

project(sgemm)
add_executable(sgemm sgemm.cxx kernels.cu)
# set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm cudart cuda cublas)
add_executable(wmma_gemm wmma_gemm.cu )
set_target_properties(wmma_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(wmma_gemm cudart cuda cublas)

# target_link_libraries(sumArray "-lcublas")
