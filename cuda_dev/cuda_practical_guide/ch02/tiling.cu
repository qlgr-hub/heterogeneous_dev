#include "../common/common.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>



__device__  int   a[256][512][512];
__device__  float b[256][512][512];


__global__ void grid3D(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t id) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // find
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // (x, y, z)
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;

    uint32_t array_size = nx * ny * nz;
    uint32_t block_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t grid_size = gridDim.x * gridDim.y * gridDim.z;
    uint32_t total_threads = block_size * grid_size;

    uint32_t thread_rank_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    uint32_t block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    uint32_t thread_rank_in_grid = block_rank_in_grid * block_size + thread_rank_in_block;

    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf(float(a[z][y][x]));
    if (thread_rank_in_grid == id) {
        printf("array size  %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
        printf("thread block  %3d x %3d x %3d = %d\n", blockDim.x, blockDim.y, blockDim.z, block_size);
        printf("thread grid  %3d x %3d x %3d = %d\n", gridDim.x, gridDim.y, gridDim.z, grid_size);
        printf("total number of threads in grid %d\n", total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
        printf("for thread with 3D-rank %d 1D-rank %d block rank in grid %d\n",
            thread_rank_in_grid, thread_rank_in_block, block_rank_in_grid);
    }
}


__global__ void grid3D_linear(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t id) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t array_size = nx * ny * nz;
    uint32_t total_threads = gridDim.x * blockDim.x;

    uint32_t tid_start = tid;
    uint32_t pass = 0;

    while (tid < array_size) {
        uint32_t x = tid % nx;
        uint32_t y = (tid / nx) % ny;
        uint32_t z = tid / (nx * ny);

        a[z][y][x] = tid;
        b[z][y][x] = sqrtf(float(a[z][y][x]));
        if (tid == id) {
            printf("array size  %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
            printf("thread block  %3d\n", blockDim.x);
            printf("thread grid  %3d\n", gridDim.x);
            printf("total number of threads in grid %d\n", total_threads);
            printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
            printf("rank_in_block = %d rank_in_grid = %d pass %d tid offset %d\n",
                threadIdx.x, tid_start, pass, tid - tid_start);
        }
        tid += gridDim.x * blockDim.x;
        ++pass;
    }
}



int main(int argc, char* argv[]) {
    uint32_t id = (argc > 1) ? atoi(argv[1]) : 12345;
    uint32_t blocks  = (argc > 2) ? atoi(argv[2]) : 288;
    uint32_t threads = (argc > 3) ? atoi(argv[3]) : 256;

    CHECK(cudaSetDevice(0));

    // dim3 thread3d{ 32, 8, 2 };   // 32*8*2 = 512
    // dim3 block3d{ 16, 64, 128 }; // 16*64*128 = 131072
    // grid3D <<<block3d, thread3d>>> (512, 512, 256, id);

    grid3D_linear <<<blocks, threads>>> (512, 512, 256, id);

    CHECK(cudaDeviceReset());
    return 0;
}