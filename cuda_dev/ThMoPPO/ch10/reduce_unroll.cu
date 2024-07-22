#include "../common/common.h"
#include <cstdlib>
#include <ctime>
#include <cstdint>


__global__ void reduceNoBankConflict(int* idata, int* odata) {
    __shared__ int data[32][32];

    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = threadIdx.x / 32;
    uint col = threadIdx.x % 32;
    data[row][col] = idata[tid];
    __syncthreads();

    for (int stride = 16; stride > 0; stride = stride >> 1) {
        if (row < stride)
            data[row][col] += data[row + stride][col];
        __syncthreads();
    }

    for (int stride = 16; stride > 0; stride = stride >> 1) {
        if (threadIdx.x < stride)
            data[0][col] += data[0][col + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) odata[blockIdx.x] = data[0][0];
}


__global__ void reduceUnroll(int* idata, int* odata) {
    __shared__ int data[32][32];

    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = threadIdx.x / 32;
    uint col = threadIdx.x % 32;
    data[row][col] = idata[tid];
    __syncthreads();

    for (int stride = 16; stride > 0; stride = stride >> 1) {
        if (row < stride)
            data[row][col] += data[row + stride][col];
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        data[0][col] += data[0][col + 16];
        __syncthreads();
        data[0][col] += data[0][col + 8];
        __syncthreads();
        data[0][col] += data[0][col + 4];
        __syncthreads();
        data[0][col] += data[0][col + 2];
        __syncthreads();
        data[0][col] += data[0][col + 1];
        __syncthreads();
    }

    if (threadIdx.x == 0) odata[blockIdx.x] = data[0][0];
}

int reduceCPU(int* data, int size) {
    int sum = 0;
    for (int i{ 0 }; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

int main() {
    srand((unsigned)time(nullptr));

    const uint32_t size = 1 << 23;
    const uint32_t bytes = size * sizeof(int);
    int* Ic = (int*)malloc(bytes);
    for (int i{ 0 }; i < size; ++i) {
        Ic[i] = static_cast<int>(rand() % 10 + 1);
    }

    double start = seconds();
    int cpuRes = reduceCPU(Ic, size);
    double elaps = seconds() - start;
    printf("reduceCPU elaps %f sec\n", elaps);


    SelectGPUDevice(0, false);
    int* Id = nullptr;
    cudaMalloc(&Id, bytes);
    cudaMemcpy(Id, Ic, bytes, cudaMemcpyHostToDevice);

    dim3 BD{1024};
    dim3 GD{size / BD.x};
    const int oBytes = GD.x * sizeof(int);
    int* Od = nullptr;
    cudaMalloc(&Od, oBytes);
    cudaMemset(Od, 0, oBytes);
    start = seconds();
    reduceNoBankConflict <<< GD, BD >>> (Id, Od);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("reduceNoBankConflict <<< %d, %d >>> elaps %f sec\n", GD.x, BD.x, elaps);

    int* Oc = (int*)malloc(oBytes);
    memset(Oc, 0, oBytes);
    cudaMemcpy(Oc, Od, oBytes, cudaMemcpyDeviceToHost);
    int gpuRes = 0;
    for (int i{ 0 }; i < GD.x; ++i) {
        gpuRes += Oc[i];
    }
    if (cpuRes == gpuRes)
        printf("result match: %d\n", cpuRes);


    cudaMemcpy(Id, Ic, bytes, cudaMemcpyHostToDevice);
    cudaMemset(Od, 0, oBytes);
    start = seconds();
    reduceUnroll <<< GD, BD >>> (Id, Od);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("reduceUnroll <<< %d, %d >>> elaps %f sec\n", GD.x, BD.x, elaps);

    memset(Oc, 0, oBytes);
    cudaMemcpy(Oc, Od, oBytes, cudaMemcpyDeviceToHost);
    gpuRes = 0;
    for (int i{ 0 }; i < GD.x; ++i) {
        gpuRes += Oc[i];
    }
    if (cpuRes == gpuRes)
        printf("result match: %d\n", cpuRes);


    cudaFree(&Id);
    cudaFree(&Od);
    free(Ic);
    free(Oc);

    cudaDeviceReset();
    return 0;
}
