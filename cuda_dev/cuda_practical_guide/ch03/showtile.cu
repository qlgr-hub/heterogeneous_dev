#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace cg = cooperative_groups;

template <uint32_t T>
__device__ void show_tile(const char* tag, cg::thread_block_tile<T> p) {
    uint32_t rank = p.thread_rank();
    uint32_t size = p.size();
    uint32_t mrank = p.meta_group_rank();
    uint32_t msize = p.meta_group_size();

    printf("%s rank in tile %2d size %2d rank %3d num %3d net size %d\n",
        tag, rank, size, mrank, msize, msize * size);
}


__global__ void cgwarp(uint32_t id) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto warp32 = cg::tiled_partition<32>(block);
    auto warp16 = cg::tiled_partition<16>(block);
    auto warp8  = cg::tiled_partition<8> (block);

    auto tiled8 = cg::tiled_partition<8>(warp32);
    auto tiled4 = cg::tiled_partition<4>(tiled8);

    if (grid.thread_rank() == id) {
        printf("warps and subwarpss for thread %d:\n", id);
        show_tile<32>("warp32", warp32);
        show_tile<16>("warp16", warp16);
        show_tile< 8>("warp8 ",  warp8);
        show_tile< 8>("tiled8", tiled8);
        show_tile< 4>("tiled4", tiled4);
    }
}


int main(int argc, char* argv[]) {
    uint32_t id      = (argc > 1) ? atoi(argv[1]) : 12345;
    uint32_t blocks  = (argc > 2) ? atoi(argv[2]) : 36000;
    uint32_t threads = (argc > 3) ? atoi(argv[3]) :   256;

    cgwarp <<<blocks,  threads>>> (id);
    cudaDeviceSynchronize();
    return 0;
}