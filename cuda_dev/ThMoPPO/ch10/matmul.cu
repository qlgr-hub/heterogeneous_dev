#include "../common/common.h"
#include <ctime>

__global__ void MatrixMulGPU(const float* Ad, const float* Bd, float* Cd, int width) {
    uint offset = threadIdx.x;
    int row = offset / width;
    int col = offset & (width - 1);

    float sum = 0.0f;
    for (int i{ 0 }; i < width; ++i) {
        sum += Ad[row * width + i] * Bd[i * width + col];
    }
    Cd[row * width + col] = sum;
}

void MatrixMulCPU(const float* A, const float* B, float* C, int width) {
    for (int i{ 0 }; i < width; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            float sum = 0.0f;
            for (int k{ 0 }; k < width; ++k) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

static bool chechResult(const float* RCPU, const float* RGPU, int size) {
    constexpr double epsilon = 1.0E-8;

    bool match = true;
    for (int i{ 0 }; i < size; ++i) {
        if (std::fabs(RCPU[i] - RGPU[i]) > epsilon) {
            match = false;
            printf("Matrixs do not match!\n");
            printf("cpu: %5.2f, gpu: %5.2f at current %d\n", RCPU[i], RGPU[i], i);
            break;
        }
    }

    if (match)
        printf("Matrixs match!\n");

    return match;
}

int main() {
    srand((unsigned)time(nullptr));
    const int Width = 1 << 5;
    const int bytes = Width * Width * sizeof(float);

    // allocate and init host side input data
    float* Ac = (float*)malloc(bytes);
    float* Bc = (float*)malloc(bytes);
    for (int i{ 0 }; i < Width; ++i) {
        for (int j{ 0 }; j < Width; ++j) {
            Ac[i * Width + j] = static_cast<float>(rand() % 10 + 1);
            Bc[i * Width + j] = static_cast<float>(rand() % 10 + 1);
        }
    }

    // compute matmul on CPU
    float* Cc = (float*)malloc(bytes);
    memset(Cc, 0, bytes);
    double start = seconds();
    MatrixMulCPU(Ac, Bc, Cc, Width);
    double elaps = seconds() - start;
    printf("MatrixMulCPU elaps %f sec\n", elaps);


    SelectGPUDevice(0);
    // allocate and init device side input data
    float* Ag = nullptr;
    float* Bg = nullptr;
    cudaMalloc((void**)&Ag, bytes);
    cudaMalloc((void**)&Bg, bytes);
    cudaMemcpy(Ag, Ac, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Bg, Bc, bytes, cudaMemcpyHostToDevice);

    // compute matmul on GPU
    float* Cg = nullptr;
    cudaMalloc((void**)&Cg, bytes);
    cudaMemset(Cg, 0, bytes);
    start = seconds();
    dim3 bDim{Width*Width};
    dim3 gDim{1};
    MatrixMulGPU <<< gDim, bDim >>> (Ag, Bg, Cg, Width);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("MatrixMulGPU <<< %d, %d >>> elaps %f sec\n", gDim.x, bDim.x, elaps);

    // check result
    float* Ct = (float*)malloc(bytes);
    memset(Ct, 0, bytes);
    cudaMemcpy(Ct, Cg, bytes, cudaMemcpyDeviceToHost);
    chechResult(Cc, Ct, Width * Width);


    // free resources
    cudaFree(&Ag);
    cudaFree(&Bg);
    cudaFree(&Cg);
    free(Ac);
    free(Bc);
    free(Cc);
    free(Ct);
    return 0;
}
