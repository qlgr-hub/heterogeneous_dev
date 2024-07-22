#include <complex>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>
#include "kernels.hxx"

static void randomInit(float* D, int sz) {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    srand(static_cast<unsigned>(tp.tv_usec));

    for (int i = 0; i < sz; ++i) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        D[i] = tmp;
    }
}

static void zeroInit(float* D, int sz) {
    for (int i = 0; i < sz; ++i) {
        D[i] = 0.0f;
    }
}

static void gemm_cpu(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int x = 0; x < M; ++x) {
        for (int y = 0; y < N; ++y) {
            float tmp = 0.0;
            for (int k = 0; k < K; ++k) {
                tmp += A[x * K + k] * B[k * N + y];
            }
            // C = α*(A@B)+β*C
            C[x * N + y] = alpha * tmp + beta * C[x * N + y];
        }
    }
}

static void checkResult(const float* R1, const float* R2, int size, bool promptMatch = false) {
    constexpr double epsilon = 0.01;

    bool match = true;
    for (int i{ 0 }; i < size; ++i) {
        if (std::fabs(R1[i] - R2[i]) > epsilon) {
            match = false;
            printf("Matrices do not match!\n");
            printf("result1: %5.2f, result2: %5.2f at current %d\n", R1[i], R2[i], i);
            break;
        }
    }

    if (promptMatch && match)
        printf("Matrices match!\n");
}


int main() {
    auto seconds = []() {
        struct timeval tp;
        gettimeofday(&tp, nullptr);
        return ( static_cast<double>(tp.tv_sec) + static_cast<double>(tp.tv_usec*1.e-6) );
    };


    const int alpha = 1;
    const int beta = 0;
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    float* A  = (float*)malloc(M * K * sizeof(float));
    float* B  = (float*)malloc(K * N * sizeof(float));
    float* C1 = (float*)malloc(M * N * sizeof(float));
    float* C2 = (float*)malloc(M * N * sizeof(float));
    randomInit(A, M * K);
    randomInit(B, K * N);
    zeroInit(C1, M * N);


    // CPU compute
    double dStart = seconds();
    // gemm_cpu(M, N, K, alpha, A, B, beta, C1);
    double dElaps = seconds() - dStart;
    // printf("gemm_cpu elapsed %f sec\n", dElaps);


    // set up GPU and compute
    preprocess(0);

    // zeroInit(C2, M * N);
    double elaps = runKernel(KType::KT_NAIVE, M, N, K, alpha, A, B, beta, C1);
    // checkResult(C1, C2, M * N);
    //printf("naive kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_GMEM_COALESCING, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("gmem coalescing kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_CACHING, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem caching kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_1D_BLOCKTILING, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem 1D blocktiling kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_2D_BLOCKTILING, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem 2D blocktiling kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_VECTORIZE, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem vectorize kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_RESOLVE_BANKCONFLICTS_1, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem resolve bank conflicts 1 kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_RESOLVE_BANKCONFLICTS_2, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem resolve bank conflicts 2 kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_SMEM_WARPTILING, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem warp tiling kernel elapsed %f sec\n", elaps);

    zeroInit(C2, M * N);
    elaps = runKernel(KType::KT_CUBLAS, M, N, K, alpha, A, B, beta, C2);
    checkResult(C1, C2, M * N);
    // printf("smem cublasGemmEx elapsed %f sec\n", elaps);
    
    free(A);
    free(B);
    free(C1);
    free(C2);

    postprocess();
    return 0;
}
