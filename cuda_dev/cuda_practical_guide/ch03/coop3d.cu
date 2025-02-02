#include "../common/common.h"
#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ int   a[256][512][512];
__device__ float b[256][512][512];

__global__ void coop3d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t id) {
    auto grid    =  cg::this_grid();
    auto block = cg::this_thread_block();

    uint32_t x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    uint32_t y = block.group_index().y * block.group_dim().y + block.thread_index().y;
    uint32_t z = block.group_index().z * block.group_dim().z + block.thread_index().z;
    if (x >= nx || y >= ny || z >= nz)
        return;

    uint32_t array_size = nx * ny * nz;
    uint32_t block_size = block.size();
    uint32_t grid_size = grid.size() / block.size();
    uint32_t total_threads = grid.size();

    uint32_t thread_rank_in_block = block.thread_rank();
    uint32_t block_rank_in_grid = grid.thread_rank() / block.size();
    uint32_t thread_rank_in_grid = grid.thread_rank();

    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf(float(a[z][y][x]));
    if (thread_rank_in_grid == id) {
        printf("array size  %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
        printf("thread block  %3d x %3d x %3d = %d\n", block.group_dim().x, block.group_dim().y, block.group_dim().z, block_size);
        printf("thread grid  %3d x %3d x %3d = %d\n", grid.dim_blocks().x, grid.dim_blocks().y, grid.dim_blocks().z, grid_size);
        printf("total number of threads in grid %d\n", total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
        printf("for thread with 3D-rank %d 1D-rank %d block rank in grid %d\n",
            thread_rank_in_grid, thread_rank_in_block, block_rank_in_grid);
    }
}

int main(int argc, char* argv[]) {
    uint32_t id = (argc > 1) ? atoi(argv[1]) : 12345;

    CHECK(cudaSetDevice(0));
    dim3 thread3d{ 32, 8, 2 };   // 32*8*2 = 512
    dim3 block3d{ 16, 64, 128 }; // 16*64*128 = 131072
    coop3d <<<block3d, thread3d>>> (512, 512, 256, id);
    CHECK(cudaDeviceReset());
    return 0;
}
