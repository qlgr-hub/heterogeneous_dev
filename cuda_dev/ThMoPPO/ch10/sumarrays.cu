#include "../common/common.h"
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <ctime>


static void randomInit(float* arr, int size) {
    for (int i{ 0 }; i < size; ++i) {
        arr[i] = static_cast<float>(rand() & 0xFF) / 10.0f;
    }
}

static void sumArraysCPU(const float* A, const float* B, float* C, int size) {
    for (int i{ 0 }; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysGPU(const float* A, const float* B, float* C, int size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

static bool chechResult(const float* RCPU, const float* RGPU, int size) {
    constexpr double epsilon = 1.0E-8;

    bool match = true;
    for (int i{ 0 }; i < size; ++i) {
        if (std::fabs(RCPU[i] - RGPU[i]) > epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("cpu: %5.2f, gpu: %5.2f at current %d\n", RCPU[i], RGPU[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match!\n");
    return match;
}

int main() {
    srand((unsigned)time(nullptr));

    // alloc CPU resource
    const int size = 1 << 24;
    const int bytes = size * sizeof(float);
    float* cA = (float*)malloc(bytes);
    float* cB = (float*)malloc(bytes);
    randomInit(cA, size);
    randomInit(cB, size);

    // compute on CPU
    float* cC = (float*)malloc(bytes);
    memset(cC, 0, bytes);
    double start = seconds();
    sumArraysCPU(cA, cB, cC, size);
    double elaps = seconds() - start;
    printf("sumArraysCPU elaps %f sec\n", elaps);


    SelectGPUDevice(0);
    // alloc GPU resource
    float* gA = nullptr;
    float* gB = nullptr;
    float* gC = nullptr;
    cudaMalloc(&gA, bytes);
    cudaMalloc(&gB, bytes);
    cudaMalloc(&gC, bytes);

    // compute on GPU
    cudaMemcpy(gA, cA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gB, cB, bytes, cudaMemcpyHostToDevice);
    cudaMemset(gC, 0, bytes);
    dim3 bDim{1024};
    dim3 gDim{(size + bDim.x - 1) / bDim.x};
    start = seconds();
    sumArraysGPU <<< gDim, bDim >>> (gA, gB, gC, size);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("sumArraysGPU <<< %d, %d >>> elaps %f sec\n", gDim.x, bDim.x, elaps);


    // check result
    float* cCg = (float*)malloc(bytes);
    memset(cCg, 0, bytes);
    cudaMemcpy(cCg, gC, bytes, cudaMemcpyDeviceToHost);
    chechResult(cC, cCg, size);


    // free resource
    cudaFree(&gA);
    cudaFree(&gB);
    cudaFree(&gC);
    cudaDeviceReset();

    free(cA);
    free(cB);
    free(cC);
    free(cCg);
    return 0;
}
