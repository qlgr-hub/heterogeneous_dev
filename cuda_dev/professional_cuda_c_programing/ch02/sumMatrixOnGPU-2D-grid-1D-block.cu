#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = blockIdx.y;
    unsigned int idx =  iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    auto cpuSecond = []() {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
    };

    dim3 block{32};
    dim3 grid{(nx + block.x - 1)/block.x, (uint)ny};
    double dStart = cpuSecond();
    sumMatrixOnGPUMix <<< grid, block >>> (d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double dElaps = cpuSecond() - dStart;
    printf("sumMatrixOnGPU1D <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, dElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    dStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    dElaps = cpuSecond() - dStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", dElaps);

    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return (0);
}