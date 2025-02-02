#include <cstdint>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../common/utils.h"
#include "../common/common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "helper_math.h"
namespace cg = cooperative_groups;



__global__ void reduce4(int* __restrict y, const int* __restrict x, uint32_t N) {
    extern __shared__ int tsum[];

    uint32_t id = threadIdx.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    tsum[id] = 0;
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


template <uint32_t blockSize>
__global__ void reduce5(int* __restrict sums, const int* __restrict data, uint32_t n) {
    __shared__ int s[blockSize];
    uint32_t id = threadIdx.x;
    s[id] = 0;

    for (uint32_t tid = blockSize * blockIdx.x + threadIdx.x; tid < n;  tid += blockSize * gridDim.x) {
        s[id] += data[tid];
    }
    __syncthreads();

    if (blockSize > 512 && id < 512 && id + 512 < blockSize)
        s[id] += s[id + 512];
    __syncthreads();

    if (blockSize > 256 && id < 256 && id + 256 < blockSize)
        s[id] += s[id + 256];
    __syncthreads();

    if (blockSize > 128 && id < 128 && id + 128 < blockSize)
        s[id] += s[id + 128];
    __syncthreads();

    if (blockSize > 64 && id < 64 && id + 64 < blockSize)
        s[id] += s[id + 64];
    __syncthreads();

    if (id < 32) {
        s[id] += s[id + 32];
        __syncwarp();

        if (id < 16)
            s[id] += s[id + 16];
        __syncwarp();

        if (id < 8)
            s[id] += s[id + 8];
        __syncwarp();

        if (id < 4)
            s[id] += s[id + 4];
        __syncwarp();

        if (id < 2)
            s[id] += s[id + 2];
        __syncwarp();

        if (id < 1)
            s[id] += s[id + 1];
        __syncwarp();

        if (id == 0)
            sums[blockIdx.x] = s[0];
    }
}


template <uint32_t blockSize>
__global__ void reduce6(int* __restrict sums, const int* __restrict data, uint32_t n) {
    __shared__ int s[blockSize];

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    uint32_t id = block.thread_rank();
    s[id] = 0;
    for (uint32_t tid = grid.thread_rank(); tid < n; tid += grid.size())
        s[id] += data[tid];
    block.sync();

    if (blockSize > 512 && id < 512 && id + 512 < blockSize)
        s[id] += s[id + 512];
    block.sync();

    if (blockSize > 256 && id < 256 && id + 256 < blockSize)
        s[id] += s[id + 256];
    block.sync();

    if (blockSize > 128 && id < 128 && id + 128 < blockSize)
        s[id] += s[id + 128];
    block.sync();

    if (blockSize > 64 && id < 64 && id + 64 < blockSize)
        s[id] += s[id + 64];
    block.sync();

    if (id < 32) {
        s[id] += s[id + 32];
        warp.sync();

        s[id] += warp.shfl_down(s[id], 16);
        s[id] += warp.shfl_down(s[id], 8 );
        s[id] += warp.shfl_down(s[id], 4 );
        s[id] += warp.shfl_down(s[id], 2 );
        s[id] += warp.shfl_down(s[id], 1 );

        if (id == 0)
            sums[block.group_index().x] = s[0];
    }
}


__global__ void reduce7(int* __restrict sums, const int* __restrict data, uint32_t n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int v = 0;
    for (uint32_t tid = grid.thread_rank(); tid < n; tid += grid.size())
        v += data[tid];
    warp.sync();

    v += warp.shfl_down(v, 16);
    v += warp.shfl_down(v, 8 );
    v += warp.shfl_down(v, 4 );
    v += warp.shfl_down(v, 2 );
    v += warp.shfl_down(v, 1 );

    if (warp.thread_rank() == 0)
        atomicAdd(&sums[block.group_index().x], v);
}


__global__ void reduce8(int* __restrict sums, const int* __restrict data, uint32_t n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int v = 0;
    for (uint32_t tid = grid.thread_rank(); tid < n; tid += grid.size())
        v += data[tid];
    warp.sync();

    v = cg::reduce(warp, v, cg::plus<int>());

    if (warp.thread_rank() == 0)
        atomicAdd(&sums[block.group_index().x], v);
}


__global__ void reduce8_vl(int* __restrict sums, const int* __restrict data, uint32_t n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int4 v4{ 0, 0, 0, 0 };
    for (uint32_t tid = grid.thread_rank(); tid < (n / 4); tid += grid.size())
        v4 += reinterpret_cast<const int4*>(data)[tid];
    int v = v4.x + v4.y + v4.z + v4.w;
    warp.sync();

    v = cg::reduce(warp, v, cg::plus<int>());

    if (warp.thread_rank() == 0)
        atomicAdd(&sums[block.group_index().x], v);
}



int main(int argc, char* argv[]) {
    uint32_t N = (argc > 1) ? atoi(argv[1]) : 1 << 24;

    thrust::host_vector<int> x(N);

    std::default_random_engine gen{ 12345678 };
    std::uniform_int_distribution<int> iran{ 0, 10 };
    for (uint32_t k = 0; k < N; ++k) {
        x[k] = iran(gen);
    }

    Utils::TimeStats tim;
    int host_sum = 0;
    for (uint32_t k = 0; k < N; ++k) {
        host_sum += x[k];
    }
    double t1 = tim.lap_milli();

    printf("sum of %d numbers: host %d %.3f ms\n", N, host_sum, t1);

    uint32_t blocks  = (argc > 2) ? atoi(argv[2]) : 360;
    uint32_t threads = (argc > 3) ? atoi(argv[3]) : 256;

    thrust::device_vector<int> dev_x(N);
    thrust::device_vector<int> dev_y(N);
    dev_x = x; // H2D copy
    tim.reset();
    reduce4 <<<blocks, threads, threads * sizeof(int)>>> (dev_y.data().get(), dev_x.data().get(), N);
    reduce4 <<<1, blocks, blocks * sizeof(int)>>> (dev_x.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t2 = tim.lap_milli();
    int gpu_sum1 = dev_x[0]; // D2H copy

    printf("GPU reduce4 %d %.3f ms\n", gpu_sum1, t2);


    dev_x = x; // H2D copy again
    tim.reset();
    if (threads == 64)
		reduce5<64> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 128)
		reduce5<128> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 256)
		reduce5<256> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 512)
		reduce5<512> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 1024)
        reduce5<1024> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);

    reduce4 <<<1, blocks, blocks * sizeof(int)>>> (dev_x.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t3 = tim.lap_milli();
    int gpu_sum2 = dev_x[0]; // D2H copy

    printf("GPU reduce5 %d %.3f ms\n", gpu_sum2, t3);


    dev_x = x; // H2D copy again
    tim.reset();
    if (threads == 64)
		reduce6<64> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 128)
		reduce6<128> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 256)
		reduce6<256> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 512)
		reduce6<512> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
	else if (threads == 1024)
        reduce6<1024> <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);

    reduce4 <<<1, blocks, blocks * sizeof(int)>>> (dev_x.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t4 = tim.lap_milli();
    int gpu_sum3 = dev_x[0]; // D2H copy

    printf("GPU reduce6 %d %.3f ms\n", gpu_sum3, t4);


    thrust::device_vector<int> dev_y1(blocks);
    thrust::host_vector<int> zeros(blocks);
    for (auto& v : zeros) {
        v = 0;
    }
    dev_y = zeros; // H2D copy
    dev_y1 = zeros;
    dev_x = x;    // H2D copy again
    tim.reset();
    reduce7 <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
    reduce7 <<<1, blocks>>> (dev_y1.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t5 = tim.lap_milli();
    int gpu_sum4 = dev_y1[0]; // D2H copy

    printf("GPU reduce7 %d %.3f ms\n", gpu_sum4, t5);


    dev_y = zeros; // H2D copy
    dev_y1 = zeros;
    dev_x = x;    // H2D copy again
    tim.reset();
    reduce8 <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
    reduce8 <<<1, blocks>>> (dev_y1.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t6 = tim.lap_milli();
    int gpu_sum5 = dev_y1[0]; // D2H copy

    printf("GPU reduce8 %d %.3f ms\n", gpu_sum5, t6);


    dev_y = zeros; // H2D copy
    dev_y1 = zeros;
    dev_x = x;    // H2D copy again
    tim.reset();
    reduce8_vl <<<blocks, threads>>> (dev_y.data().get(),dev_x.data().get(),N);
    reduce8_vl <<<1, blocks>>> (dev_y1.data().get(), dev_y.data().get(), blocks);
    CHECK(cudaDeviceSynchronize());
    double t7 = tim.lap_milli();
    int gpu_sum6 = dev_y1[0]; // D2H copy

    printf("GPU reduce8 %d %.3f ms [vl]\n", gpu_sum6, t7);
    return 0;
}
