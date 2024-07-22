#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) \
{\
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    {\
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }\
}

__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }

    c [tid] = a + b;
}

__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ( (tid / warpSize) % 2 == 0 ) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (tid % 2 == 0);
    if ( ipred ) {
        a = 100.0f;
    }
    if (!ipred) {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size      = atoi(argv[2]);
    printf("Data size %d \n", size);

    // set up execution configuration
    dim3 block{(uint)blocksize, 1};
    dim3 grid{(size + block.x - 1) / block.x, 1};
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // run a warmup kernel to remove overhead
    double iStart, iElaps;
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("warmup      <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 1
    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 3
    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 3
    iStart = seconds();
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps);
    CHECK(cudaGetLastError());

    // run kernel 4
    iStart = seconds();
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel4 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps);
    CHECK(cudaGetLastError());

    // free gpu memory and reset divece
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}