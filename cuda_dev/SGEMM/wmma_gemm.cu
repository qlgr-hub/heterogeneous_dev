#include <fstream>
#include <iomanip>
#include <ios>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <cublas_v2.h>

#define CHECK(err) _checkCudaRTCall(err, __FILE__, __LINE__)

static inline void _checkCudaRTCall(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) { 
        fprintf(stderr, "Error: %s:%d, ", file, line); 
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err));
    }
}

static inline unsigned CEIL_DIV(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}


__global__ void gemm_naive(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float tmp = 0.0f;

        for (int k{ 0 }; k < K; ++k) {
            tmp += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        }

        // C = α*(A@B)+β*C
        C[i * N + j] = alpha * tmp + beta * C[i * N + j];
    }
}

void launch_naive(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    dim3 blockDim{ 16, 16, 1 };
    dim3 gridDim{ CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.x), 1 };

    gemm_naive <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


// The only dimensions currently supported by WMMA
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16; 
__global__ void wmma_naive(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    using namespace nvcuda;
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i{ 0 }; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * K + bCol, K);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);

        // #pragma unroll
        for(int i{ 0 }; i < c_frag.num_elements; ++i) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

static constexpr uint WARPSIZE = 32;
void launch_wmma_naive(int M, int N, int K, float alpha, const half* A, half* B, float beta, float* C) {
    dim3 blockDim{ 128, 4 };
    dim3 gridDim;
    gridDim.x = (M + (WMMA_M * blockDim.x / WARPSIZE - 1)) / (WMMA_M * blockDim.x / WARPSIZE);
    gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
 
    // printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    wmma_naive <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


void launch_culas_half(cublasHandle_t handle, int M, int N, int K, float alpha, const half* A, half* B, float beta, float* C) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, B, CUDA_R_16F, N,
                A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, 
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK(cudaDeviceSynchronize());
}


static void randomInit(half* D, int sz) {
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

// static void gemm_cpu(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
//     for (int x = 0; x < M; ++x) {
//         for (int y = 0; y < N; ++y) {
//             float tmp = 0.0;
//             for (int k = 0; k < K; ++k) {
//                 tmp += (float)A[x * K + k] * (float)B[k * N + y];
//             }
//             // C = α*(A@B)+β*C
//             C[x * N + y] = alpha * tmp + beta * C[x * N + y];
//         }
//     }
// }

static void checkResult(const float* R1, const float* R2, int size, bool promptMatch = false) {
    constexpr double epsilon = 1e-1;

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


    constexpr int alpha = 1;
    constexpr int beta = 0;
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int bytesA = M * K * sizeof(half);
    constexpr int bytesB = K * N * sizeof(half);
    constexpr int bytesC = M * N * sizeof(float);

    half* A  = (half*)malloc(bytesA);
    half* B  = (half*)malloc(bytesB);
    float* C1 = (float*)malloc(bytesC);
    float* C2 = (float*)malloc(bytesC);
    randomInit(A, M * K);
    randomInit(B, K * N);
    zeroInit(C1, M * N);


    // CPU compute
    double dStart = seconds();
    // gemm_cpu(M, N, K, alpha, A, B, beta, C1);
    double dElaps = seconds() - dStart;
    // printf("gemm_cpu elapsed %f sec\n", dElaps);


    // set up GPU and compute
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));

    // allocate device side resources
    half* Ad = nullptr;
    half* Bd = nullptr;
    float* Cd = nullptr;
    CHECK(cudaMalloc(&Ad, bytesA));
    CHECK(cudaMalloc(&Bd, bytesB));
    CHECK(cudaMemcpy(Ad, A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Bd, B, bytesB, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&Cd, bytesC));
    CHECK(cudaMemset(Cd, 0, bytesC));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    // device wramup (doesn't work)
    // launch_naive(M, N, K, alpha, Ad, Bd, beta, Cd);
    // CHECK(cudaDeviceSynchronize());


    // launch naive gemm
    //zeroInit(C2, M * N);
    // dStart = seconds();
    cudaEventRecord(beg);
    launch_naive(M, N, K, alpha, Ad, Bd, beta, Cd);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elaps{ 0.f };
    cudaEventElapsedTime(&elaps, beg, end);
    // CHECK(cudaDeviceSynchronize());
    // dElaps = seconds() - dStart;
    printf("gemm gpu naive elapsed %f sec\n", elaps / 1000.);
    // return result
    CHECK(cudaMemcpy(C1, Cd, bytesC, cudaMemcpyDeviceToHost));
    // checkResult(C1, C2, M * N);


    // test wramup (doesn't work)
    // CHECK(cudaMemset(Cd, 0, bytesC));
    // launch_wmma_naive(M, N, K, alpha, Ad, Bd, beta, Cd);
    // launch wmma naive gemm
    CHECK(cudaMemset(Cd, 0, bytesC));
    zeroInit(C2, M * N);
    // dStart = seconds();
    cudaEventRecord(beg);
    launch_wmma_naive(M, N, K, alpha, Ad, Bd, beta, Cd);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    elaps = 0.f;
    cudaEventElapsedTime(&elaps, beg, end);
    // CHECK(cudaDeviceSynchronize());
    // dElaps = seconds() - dStart;
    printf("gemm gpu wmma naive elapsed %f sec\n", elaps / 1000.);
    CHECK(cudaMemcpy(C2, Cd, bytesC, cudaMemcpyDeviceToHost));
    checkResult(C1, C2, M * N);


    // launch cublas gemm
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    }

    // cuBLAS wramup
    CHECK(cudaMemset(Cd, 0, bytesC));
    launch_culas_half(handle, M, N, K, alpha, Ad, Bd, beta, Cd);

    CHECK(cudaMemset(Cd, 0, bytesC));
    // dStart = seconds();
    cudaEventRecord(beg);
    launch_culas_half(handle, M, N, K, alpha, Ad, Bd, beta, Cd);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    elaps = 0.f;
    cudaEventElapsedTime(&elaps, beg, end);
    // CHECK(cudaDeviceSynchronize());
    // dElaps = seconds() - dStart;
    printf("gemm gpu cublas elapsed %f sec\n", elaps / 1000.);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(C2, Cd, bytesC, cudaMemcpyDeviceToHost));
    checkResult(C1, C2, M * N);


    cudaFree(&Ad);
    cudaFree(&Bd);
    cudaFree(&Cd);
    free(A);
    free(B);
    free(C1);
    free(C2);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cublasDestroy(handle);

    cudaDeviceReset();
    return 0;
}
