#include "../common/common.h"
#include <ctime>



__global__ void MatrixMulGPU_Shared(const float* Ad, const float* Bd, float* Cd, int width) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    uint index_j = row * 4;
    uint index_i = col * 4;
    uint index_Cd = index_j * width + index_i;

    uint offset_inner = threadIdx.y * blockDim.x + threadIdx.x;
    uint ldsm_row = blockIdx.y * blockDim.y * 4 + (offset_inner % 128);
    uint ldsm_col = offset_inner / 128;
    uint ldsn_row = offset_inner / 128;
    uint ldsn_col = blockIdx.x * blockDim.x * 4 + (offset_inner % 128);
    float rA[4], rB[4], rC[16];

    rC[0]  = 0;
    rC[1]  = 0;
    rC[2]  = 0;
    rC[3]  = 0;
    rC[4]  = 0;
    rC[5]  = 0;
    rC[6]  = 0;
    rC[7]  = 0;
    rC[8]  = 0;
    rC[9]  = 0;
    rC[10] = 0;
    rC[11] = 0;
    rC[12] = 0;
    rC[13] = 0;
    rC[14] = 0;
    rC[15] = 0;

    __shared__ float ldsa[1024];
    __shared__ float ldsb[1024];

    for (int i{ 0 }; i < width; i += 8) {
        ldsa[offset_inner] = Ad[ldsm_row * width + ldsm_col + i];
        ldsb[offset_inner] = Bd[(ldsn_row + i) * width + ldsn_col];
        __syncthreads();

        for (int j{ 0 }; j < 8; ++j) {
            rA[0] = ldsa[threadIdx.y * 4 + (j * 128) + 0];
            rA[1] = ldsa[threadIdx.y * 4 + (j * 128) + 1];
            rA[2] = ldsa[threadIdx.y * 4 + (j * 128) + 2];
            rA[3] = ldsa[threadIdx.y * 4 + (j * 128) + 3];

            rB[0] = ldsb[threadIdx.x * 4 + (j * 128) + 0];
            rB[1] = ldsb[threadIdx.x * 4 + (j * 128) + 1];
            rB[2] = ldsb[threadIdx.x * 4 + (j * 128) + 2];
            rB[3] = ldsb[threadIdx.x * 4 + (j * 128) + 3];

            rC[0]  = rC[0]  + rA[0] * rB[0];
            rC[1]  = rC[1]  + rA[0] * rB[1];
            rC[2]  = rC[2]  + rA[0] * rB[2];
            rC[3]  = rC[3]  + rA[0] * rB[3];
            rC[4]  = rC[4]  + rA[1] * rB[0];
            rC[5]  = rC[5]  + rA[1] * rB[1];
            rC[6]  = rC[6]  + rA[1] * rB[2];
            rC[7]  = rC[7]  + rA[1] * rB[3];
            rC[8]  = rC[8]  + rA[2] * rB[0];
            rC[9]  = rC[9]  + rA[2] * rB[1];
            rC[10] = rC[10] + rA[2] * rB[2];
            rC[11] = rC[11] + rA[2] * rB[3];
            rC[12] = rC[12] + rA[3] * rB[0];
            rC[13] = rC[13] + rA[3] * rB[1];
            rC[14] = rC[14] + rA[3] * rB[2];
            rC[15] = rC[15] + rA[3] * rB[3];
        }
        __syncthreads();
    }

    Cd[index_Cd + 0 * width + 0] = rC[0];
    Cd[index_Cd + 0 * width + 1] = rC[1];
    Cd[index_Cd + 0 * width + 2] = rC[2];
    Cd[index_Cd + 0 * width + 3] = rC[3];
    Cd[index_Cd + 1 * width + 0] = rC[4];
    Cd[index_Cd + 1 * width + 1] = rC[5];
    Cd[index_Cd + 1 * width + 2] = rC[6];
    Cd[index_Cd + 1 * width + 3] = rC[7];
    Cd[index_Cd + 2 * width + 0] = rC[8];
    Cd[index_Cd + 2 * width + 1] = rC[9];
    Cd[index_Cd + 2 * width + 2] = rC[10];
    Cd[index_Cd + 2 * width + 3] = rC[11];
    Cd[index_Cd + 3 * width + 0] = rC[12];
    Cd[index_Cd + 3 * width + 1] = rC[13];
    Cd[index_Cd + 3 * width + 2] = rC[14];
    Cd[index_Cd + 3 * width + 3] = rC[15];
}

__global__ void MatrixMulGPU_4x4(const float* Ad, const float* Bd, float* Cd, int width) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    uint index_j = row * 4;
    uint index_i = col * 4;
    uint index_Cd = index_j * width + index_i;

    float rA[4], rB[4], rC[16];

    rC[0]  = 0;
    rC[1]  = 0;
    rC[2]  = 0;
    rC[3]  = 0;
    rC[4]  = 0;
    rC[5]  = 0;
    rC[6]  = 0;
    rC[7]  = 0;
    rC[8]  = 0;
    rC[9]  = 0;
    rC[10] = 0;
    rC[11] = 0;
    rC[12] = 0;
    rC[13] = 0;
    rC[14] = 0;
    rC[15] = 0;

    for (int i{ 0 }; i < width; ++i) {
        rA[0] = Ad[(index_j + 0) * width + i];
        rA[1] = Ad[(index_j + 1) * width + i];
        rA[2] = Ad[(index_j + 2) * width + i];
        rA[3] = Ad[(index_j + 3) * width + i];

        rB[0] = Bd[i * width + index_i + 0];
        rB[1] = Bd[i * width + index_i + 1];
        rB[2] = Bd[i * width + index_i + 2];
        rB[3] = Bd[i * width + index_i + 3];

        rC[0]  = rC[0]  + rA[0] * rB[0];
        rC[1]  = rC[1]  + rA[0] * rB[1];
        rC[2]  = rC[2]  + rA[0] * rB[2];
        rC[3]  = rC[3]  + rA[0] * rB[3];
        rC[4]  = rC[4]  + rA[1] * rB[0];
        rC[5]  = rC[5]  + rA[1] * rB[1];
        rC[6]  = rC[6]  + rA[1] * rB[2];
        rC[7]  = rC[7]  + rA[1] * rB[3];
        rC[8]  = rC[8]  + rA[2] * rB[0];
        rC[9]  = rC[9]  + rA[2] * rB[1];
        rC[10] = rC[10] + rA[2] * rB[2];
        rC[11] = rC[11] + rA[2] * rB[3];
        rC[12] = rC[12] + rA[3] * rB[0];
        rC[13] = rC[13] + rA[3] * rB[1];
        rC[14] = rC[14] + rA[3] * rB[2];
        rC[15] = rC[15] + rA[3] * rB[3];
    }

    Cd[index_Cd + 0 * width + 0] = rC[0];
    Cd[index_Cd + 0 * width + 1] = rC[1];
    Cd[index_Cd + 0 * width + 2] = rC[2];
    Cd[index_Cd + 0 * width + 3] = rC[3];
    Cd[index_Cd + 1 * width + 0] = rC[4];
    Cd[index_Cd + 1 * width + 1] = rC[5];
    Cd[index_Cd + 1 * width + 2] = rC[6];
    Cd[index_Cd + 1 * width + 3] = rC[7];
    Cd[index_Cd + 2 * width + 0] = rC[8];
    Cd[index_Cd + 2 * width + 1] = rC[9];
    Cd[index_Cd + 2 * width + 2] = rC[10];
    Cd[index_Cd + 2 * width + 3] = rC[11];
    Cd[index_Cd + 3 * width + 0] = rC[12];
    Cd[index_Cd + 3 * width + 1] = rC[13];
    Cd[index_Cd + 3 * width + 2] = rC[14];
    Cd[index_Cd + 3 * width + 3] = rC[15];
}

__global__ void MatrixMulGPU_2DGrid2DBlock(const float* Ad, const float* Bd, float* Cd, int width) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int Width = 1 << 12;
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
    // MatrixMulCPU(Ac, Bc, Cc, Width);
    double elaps = seconds() - start;
    // printf("MatrixMulCPU elaps %f sec\n", elaps);


    SelectGPUDevice(0, false);
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
    dim3 bDim{32, 32};
    dim3 gDim{32, 32};
    MatrixMulGPU_4x4 <<< gDim, bDim >>> (Ag, Bg, Cg, Width);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("MatrixMulGPU_4x4 <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", gDim.x, gDim.y, bDim.x, bDim.y, elaps);

    // check result
    float* Ct = (float*)malloc(bytes);
    memset(Ct, 0, bytes);
    cudaMemcpy(Ct, Cg, bytes, cudaMemcpyDeviceToHost);
    // chechResult(Cc, Ct, Width * Width);


    // compute matmul on GPU 2
    cudaMemset(Cg, 0, bytes);
    start = seconds();
    dim3 bDim1{64, 16};
    dim3 gDim1{Width / bDim1.x, Width / bDim1.y};
    MatrixMulGPU_2DGrid2DBlock <<< gDim1, bDim1 >>> (Ag, Bg, Cg, Width);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("MatrixMulGPU_2DGrid2DBlock <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", gDim1.x, gDim1.y, bDim1.x, bDim1.y, elaps);

    // check result
    cudaMemcpy(Cc, Cg, bytes, cudaMemcpyDeviceToHost);
    chechResult(Cc, Ct, Width * Width);


    cudaMemset(Cg, 0, bytes);
    start = seconds();
    dim3 bDim2{32, 32};
    dim3 gDim2{32, 32};
    MatrixMulGPU_Shared <<< gDim2, bDim2 >>> (Ag, Bg, Cg, Width);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("MatrixMulGPU_Shared <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", gDim2.x, gDim2.y, bDim2.x, bDim2.y, elaps);

    // check result
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

    cudaDeviceReset();
    return 0;
}
