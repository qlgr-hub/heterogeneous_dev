#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../common/utils.h"
#include "../common/common.h"


__global__ void reduce0(float* x, uint32_t m) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}

__global__ void reduce1(float* x, uint32_t N) {
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

	float tsum = 0.f;
	for(uint32_t k = tid; k < N; k += stride) {
        tsum += x[k];
    }
	x[tid] = tsum;
}


__global__ void reduce2(float* y, float* x, uint32_t N) {
    extern __shared__ float tsum[];

    uint32_t id = threadIdx.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    tsum[id] = 0.f;
    for(uint32_t k = tid; k < N; k += stride) {
        tsum[id] += x[k];
    }
    __syncthreads();

    for (uint32_t k = blockDim.x / 2; k > 0; k /= 2) {
        if (id < k) {
            tsum[id] += tsum[id + k];
        }
        __syncthreads();
    }

    if (id == 0) {
        y[blockIdx.x] = tsum[0];
    }
}


__device__ int pow2ceil(int n) {
    int pow2 = 1 << (31 - __clz(n));
    if(n > pow2) pow2 = (pow2 << 1);
    return pow2;
}

__global__ void reduce3(float* y, float* x, uint32_t N) {
    extern __shared__ float tsum[];

    uint32_t id = threadIdx.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    tsum[id] = 0.f;
    for(uint32_t k = tid; k < N; k += stride) {
        tsum[id] += x[k];
    }
    __syncthreads();

    uint32_t block2 = pow2ceil(blockDim.x);
    for (uint32_t k = block2 / 2; k > 0; k /= 2) {
        if (id < k && id + k < blockDim.x) {
            tsum[id] += tsum[id + k];
        }
        __syncthreads();
    }

    if (id == 0) {
        y[blockIdx.x] = tsum[0];
    }
}


__global__ void reduce4(float* y, float* x, uint32_t N) {
    extern __shared__ float tsum[];

    uint32_t id = threadIdx.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    tsum[id] = 0.f;
    for(uint32_t k = tid; k < N; k += stride) {
        tsum[id] += x[k];
    }
    __syncthreads();

    if (id < 256 && id + 256 < blockDim.x)
        tsum[id] += tsum[id + 256];
    __syncthreads();
    if (id < 128)
        tsum[id] += tsum[id + 128];
    __syncthreads();
    if (id < 64)
        tsum[id] += tsum[id + 64];
    __syncthreads();
    if (id < 32)
        tsum[id] += tsum[id + 32];
    __syncthreads();

    if (id < 16)
        tsum[id] += tsum[id + 16];
    __syncwarp();
    if (id < 8)
        tsum[id] += tsum[id + 8];
    __syncwarp();
    if (id < 4)
        tsum[id] += tsum[id + 4];
    __syncwarp();
    if (id < 2)
        tsum[id] += tsum[id + 2];
    __syncwarp();

    if (id == 0) {
        y[blockIdx.x] = tsum[0] + tsum[1];
    }
}



int main(int argc, char* argv[]) {
    uint32_t N = (argc > 1) ? atoi(argv[1]) : 1 << 24;

    thrust::host_vector<float>       x(N);
    thrust::device_vector<float> dev_x(N);

    std::default_random_engine gen{ 12345678 };
    std::uniform_real_distribution<float> fran{ 0.f, 1.f };
    for (uint32_t k = 0; k < N; ++k) {
        x[k] = fran(gen);
    }
    dev_x = x; // H2D copy

    Utils::TimeStats tim;
    double host_sum = 0.0;
    for (uint32_t k = 0; k < N; ++k) {
        host_sum += x[k];
    }
    double t1 = tim.lap_sec();

    tim.reset();
    uint32_t min_block_size = 256;
    uint32_t min_grid_size = 1;
    for (uint32_t m = N / 2; m > 0; m /= 2) {
        uint32_t threads = std::min(min_block_size, m);
        uint32_t blocks = std::max(m / min_block_size, min_grid_size);
        reduce0 <<<blocks, threads>>> (dev_x.data().get(), m);
    }
    CHECK(cudaDeviceSynchronize());
    double t2 = tim.lap_sec();

    double gpu_sum1 = dev_x[0]; // D2H copy

    dev_x = x; // H2D copy again
    tim.reset();
    uint32_t threads = 256;
    uint32_t blocks = 360;
    reduce1 <<<blocks, threads>>> (dev_x.data().get(), N);
    reduce1 <<<1, threads>>> (dev_x.data().get(), blocks * threads);
    reduce1 <<<1, 1>>> (dev_x.data().get(), threads);
    CHECK(cudaDeviceSynchronize());
    double t3 = tim.lap_sec();

    double gpu_sum2 = dev_x[0]; // D2H copy again

    // The calculation results have been verified to be correct using int type,
    // but there are accumulated errors when using float.
    printf("sum of %d random numbers: host %.1f %.7f sec, GPU reduce0 %.1f %.7f sec"
        ", GPU reduce1 %.1f %.7f sec\n", N, host_sum, t1, gpu_sum1, t2, gpu_sum2, t3);



    // use command line argument
    blocks  = (argc > 2) ? atoi(argv[2]) : 256;
    threads = (argc > 3) ? atoi(argv[3]) : 256;
    thrust::device_vector<float> dy(blocks);

    dev_x = x; // H2D copy again
    tim.reset();
    reduce2 <<<blocks, threads, threads * sizeof(float)>>> (dy.data().get(), dev_x.data().get(), N);
    reduce2 <<<1, blocks, blocks * sizeof(float)>>> (dev_x.data().get(), dy.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t4 = tim.lap_sec();
    double gpu_sum3 = dev_x[0]; // D2H copy again


    blocks = 360;
    dev_x = x; // H2D copy again
    tim.reset();
    reduce3 <<<blocks, threads, threads * sizeof(float)>>> (dy.data().get(), dev_x.data().get(), N);
    reduce3 <<<1, blocks, blocks * sizeof(float)>>> (dev_x.data().get(), dy.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t5 = tim.lap_sec();
    double gpu_sum4 = dev_x[0]; // D2H copy again


    dev_x = x; // H2D copy again
    tim.reset();
    reduce4 <<<blocks, threads, threads * sizeof(float)>>> (dy.data().get(), dev_x.data().get(), N);
    reduce4 <<<1, blocks, blocks * sizeof(float)>>> (dev_x.data().get(), dy.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t6 = tim.lap_sec();
    double gpu_sum5 = dev_x[0]; // D2H copy again

    printf("GPU reduce2 %.1f %.7f sec, GPU reduce3 %.1f %.7f sec, GPU reduce4 %.1f %.7f sec\n",
        gpu_sum3, t4, gpu_sum4, t5, gpu_sum5, t6);
    return 0;
}