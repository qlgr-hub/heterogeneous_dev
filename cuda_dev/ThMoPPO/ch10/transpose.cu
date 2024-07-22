#include "../common/common.h"
#include <complex>
#include <cstdlib>
#include <ctime>

__global__ void transpose1(float* Ad, float* Bd, int width) {
    uint nx = blockIdx.x * blockDim.x + threadIdx.x;
    uint ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < width && ny < width)
        Bd[nx * width + ny] = Ad[ny * width + nx];
}

__global__ void transpose2(float* Ad, float* Bd, int width) {
    uint nx = blockIdx.x * blockDim.x + threadIdx.x;
    uint ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < width && ny < width)
        Bd[ny * width + nx] = Ad[nx * width + ny];
}

static void transposeCPU(float* A, float* B, int width) {
    for (int i{ 0 }; i < width; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            B[i * width + j] = A[j * width + i];
        }
    }
}

static void checkResult(float* A, float* B, int size) {
    constexpr double EPSILON = 1.0E-8;

    bool match = true;
    for (int i{ 0 }; i < size; ++i) {
        if (std::fabs(A[i] - B[i]) > EPSILON) {
            match = false;
            printf("results are not match: ");
            printf("pos: %d => result1=%f, result2=%f\n", i, A[i], B[i]);
            break;
        }
    }

    if (match)
        printf("results are match!\n");
}

int main() {
    srand((unsigned)time(nullptr));
    const int Width = 4096;
    const int bytes = Width * Width * sizeof(float);

    float* Ah = (float*)malloc(bytes);
    for (int i{ 0 }; i < Width; ++i) {
        for (int j{ 0 }; j < Width; ++j) {
            Ah[i * Width + j] = static_cast<float>(rand() % 10 + 1);
        }
    }

    float* Bh = (float*)malloc(bytes);
    memset(Bh, 0, bytes);
    transposeCPU(Ah, Bh, Width);

    float* Ad = nullptr;
    float* Bd = nullptr;
    cudaMalloc(&Ad, bytes);
    cudaMemcpy(Ad, Ah, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Bd, bytes);
    cudaMemset(Bd, 0, bytes);
    dim3 BD{32, 32};
    dim3 GD{Width / 32, Width / 32};
    double start = seconds();
    transpose1 <<< GD, BD >>> (Ad, Bd, Width);
    CHECK(cudaDeviceSynchronize());
    double elaps = seconds() - start;
    printf("transpose1 <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", GD.x, GD.y, BD.x, BD.y, elaps);

    float* Bd_h = (float*)malloc(bytes);
    memset(Bd_h, 0, bytes);
    cudaMemcpy(Bd_h, Bd, bytes, cudaMemcpyDeviceToHost);
    checkResult(Bh, Bd_h, Width * Width);


    cudaMemset(Bd, 0, bytes);
    start = seconds();
    transpose2 <<< GD, BD >>> (Ad, Bd, Width);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("transpose2 <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", GD.x, GD.y, BD.x, BD.y, elaps);

    memset(Bd_h, 0, bytes);
    cudaMemcpy(Bd_h, Bd, bytes, cudaMemcpyDeviceToHost);
    checkResult(Bh, Bd_h, Width * Width);

    cudaFree(&Ad);
    cudaFree(&Bd);
    free(Ah);
    free(Bh);
    free(Bd_h);

    cudaDeviceReset();
    return 0;
}
