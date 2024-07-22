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
add_executable(sumarrays sumarrays.cu ../common/common.cu)
set_target_properties(sumarrays PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(matmul matmul.cu ../common/common.cu)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(matmul_multiblock matmul_multiblock.cu ../common/common.cu)
set_target_properties(matmul_multiblock PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(reduce reduce.cu ../common/common.cu)
set_target_properties(reduce PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(matmul_4x4 matmul_4x4.cu ../common/common.cu)
set_target_properties(matmul_4x4 PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(transpose transpose.cu ../common/common.cu)
set_target_properties(transpose PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(matmul_shared matmul_shared.cu ../common/common.cu)
set_target_properties(matmul_shared PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(reduce_unroll reduce_unroll.cu ../common/common.cu)
set_target_properties(reduce_unroll PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

# target_link_libraries(sumArray "-lcublas")
