#include "kernels.hxx"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include <mma.h>
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

// static inline double seconds() {
//     struct timeval tp;
//     gettimeofday(&tp, nullptr);
//     return ( static_cast<double>(tp.tv_sec) + static_cast<double>(tp.tv_usec*1.e-6) );
// }


__global__ void gemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float tmp = 0.0f;

        for (int k{ 0 }; k < K; ++k) {
            tmp += A[i * K + k] * B[k * N + j];
        }

        // C = α*(A@B)+β*C
        C[i * N + j] = alpha * tmp + beta * C[i * N + j];
    }
}

void launch_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    dim3 blockDim{ 32, 32, 1 };
    dim3 gridDim{ CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.x), 1 };

    gemm_naive <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BLOCKSIZE>
__global__ void gemm_gmem_coalescing(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (i < M && j < N) {
        float tmp = 0.0f;

        for (int k{ 0 }; k < K; ++k) {
            tmp += A[i * K + k] * B[k * N + j];
        }

        // C = α*(A@B)+β*C
        C[i * N + j] = alpha * tmp + beta * C[i * N + j];
    }
}

void launch_gmem_coalescing(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    dim3 blockDim{ 32 * 32, 1, 1 };
    dim3 gridDim{ CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1 };

    gemm_gmem_coalescing<32> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BLOCKSIZE>
__global__ void gemm_smem_caching(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {

    // __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    // __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    
    A += blockIdx.y * BLOCKSIZE * K;
    B += blockIdx.x * BLOCKSIZE;
    C += blockIdx.y * BLOCKSIZE * N + blockIdx.x * BLOCKSIZE;

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx{ 0 }; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

void launch_smem_caching(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    dim3 blockDim{ 32 * 32, 1, 1 };
    dim3 gridDim{ CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1 };

    gemm_smem_caching<32> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void gemm_smem_1D_blocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    const uint colAs = threadIdx.x % BK;
    const uint rowAs = threadIdx.x / BK;
    const uint colBs = threadIdx.x % BN;
    const uint rowBs = threadIdx.x / BN;

    float resultTh[TM] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        As[rowAs * BK + colAs] = A[rowAs * K + colAs];
        Bs[rowBs * BN + colBs] = B[rowBs * N + colBs];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{ 0 }; dotIdx < BK; ++dotIdx) {
            // 这个部分就明显减少了从共享内存加载数据的次数
            float tmpB = Bs[dotIdx * BN + threadCol];

            for (int resIdx{ 0 }; resIdx < TM; ++resIdx) {
                resultTh[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx{ 0 }; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * resultTh[resIdx] 
            + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

void launch_smem_1D_blocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    // same as gemm_smem_caching
    // const uint BM = 32;
    // const uint BN = 32;
    // const uint BK = 32;
    // const uint TM = 1;

    dim3 blockDim{ (BM * BN) / TM, 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_1D_blocktiling<BM, BN, BK, TM> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_smem_2D_blocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint colAs = threadIdx.x % BK;
    const uint rowAs = threadIdx.x / BK;
    const uint colBs = threadIdx.x % BN;
    const uint rowBs = threadIdx.x / BN;

    const uint threadsPerBlock = (BM * BN) / (TM * TN);
    const uint strideA = threadsPerBlock / BK;
    const uint strideB = threadsPerBlock / BN;

    float resultTh[TM * TN] = { 0.0f };
    float regA[TM] = { 0.0f };
    float regB[TN] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        for (int offset{ 0 }; offset < BM; offset += strideA) {
            As[(rowAs + offset) * BK + colAs] = A[(rowAs + offset) * K + colAs];
        }

        for (int offset{ 0 }; offset < BK; offset += strideB) {
            Bs[(rowBs + offset) * BN + colBs] = B[(rowBs + offset) * N + colBs];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{ 0 }; dotIdx < BK; ++dotIdx) {
            for (int i{ 0 }; i < TM; ++i) {
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }

            for (int i{ 0 }; i < TN; ++i) {
                regB[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for (int resM{ 0 }; resM < TM; ++resM) {
                for (int resN{ 0 }; resN < TN; ++resN) {
                    resultTh[resM * TN + resN] += regA[resM] * regB[resN];
                }
            }
        }
        __syncthreads();
    }

    for (int resM{ 0 }; resM < TM; ++resM) {
        for (int resN{ 0 }; resN < TN; ++resN) {
            C[(threadRow * TM + resM) * N + threadCol * TN + resN] = 
                alpha * resultTh[resM * TN + resN] + beta * C[(threadRow * TM + resM) * N + threadCol * TN + resN];
        }
    }
}

void launch_smem_2D_blocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 blockDim{ (BM * BN) / (TM * TN), 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_2D_blocktiling<BM, BN, BK, TM, TN> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_smem_vectorize(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint colAs = threadIdx.x % (BK / 4);
    const uint rowAs = threadIdx.x / (BK / 4);
    const uint colBs = threadIdx.x % (BN / 4);
    const uint rowBs = threadIdx.x / (BN / 4);

    float resultTh[TM * TN] = { 0.0f };
    float regA[TM] = { 0.0f };
    float regB[TN] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        float4 tmp = reinterpret_cast<const float4*>(&A[rowAs * K + colAs * 4])[0];
        As[(colAs * 4 + 0) * BM + rowAs] = tmp.x;
        As[(colAs * 4 + 1) * BM + rowAs] = tmp.y;
        As[(colAs * 4 + 2) * BM + rowAs] = tmp.z;
        As[(colAs * 4 + 3) * BM + rowAs] = tmp.w;

        reinterpret_cast<float4*>(&Bs[rowBs * BN + colBs * 4])[0] =
            reinterpret_cast<const float4*>(&B[rowBs * N + colBs * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{ 0 }; dotIdx < BK; ++dotIdx) {
            for (int i{ 0 }; i < TM; ++i) {
                regA[i] = As[dotIdx * BM + threadRow * TM + i];
            }

            for (int i{ 0 }; i < TN; ++i) {
                regB[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for (int resM{ 0 }; resM < TM; ++resM) {
                for (int resN{ 0 }; resN < TN; ++resN) {
                    resultTh[resM * TN + resN] += regA[resM] * regB[resN];
                }
            }
        }
        __syncthreads();
    }

    for (int resM{ 0 }; resM < TM; ++resM) {
        for (int resN{ 0 }; resN < TN; resN += 4) {
            float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0];
            tmp.x = alpha * resultTh[resM * TN + resN + 0] + beta * tmp.x;
            tmp.y = alpha * resultTh[resM * TN + resN + 1] + beta * tmp.y;
            tmp.z = alpha * resultTh[resM * TN + resN + 2] + beta * tmp.z;
            tmp.w = alpha * resultTh[resM * TN + resN + 3] + beta * tmp.w;
            reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0] = tmp;
        }
    }
}

void launch_smem_vectorize(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 blockDim{ (BM * BN) / (TM * TN), 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_vectorize<BM, BN, BK, TM, TN> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_smem_resolve_bankconflicts_1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint colAs = threadIdx.x % (BK / 4);
    const uint rowAs = threadIdx.x / (BK / 4);
    const uint colBs = threadIdx.x % (BN / 4);
    const uint rowBs = threadIdx.x / (BN / 4);

    float resultTh[TM * TN] = { 0.0f };
    float regA[TM] = { 0.0f };
    float regB[TN] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        float4 tmp = reinterpret_cast<const float4*>(&A[rowAs * K + colAs * 4])[0];
        As[(colAs * 4 + 0) * BM + rowAs] = tmp.x;
        As[(colAs * 4 + 1) * BM + rowAs] = tmp.y;
        As[(colAs * 4 + 2) * BM + rowAs] = tmp.z;
        As[(colAs * 4 + 3) * BM + rowAs] = tmp.w;

        tmp = reinterpret_cast<const float4*>(&B[rowBs * N + colBs * 4])[0];
        Bs[((colBs % 2) * 4 + rowBs * 8 + 0) * 16 + colBs / 2] = tmp.x;
        Bs[((colBs % 2) * 4 + rowBs * 8 + 1) * 16 + colBs / 2] = tmp.y;
        Bs[((colBs % 2) * 4 + rowBs * 8 + 2) * 16 + colBs / 2] = tmp.z;
        Bs[((colBs % 2) * 4 + rowBs * 8 + 3) * 16 + colBs / 2] = tmp.w;
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{ 0 }; dotIdx < BK; ++dotIdx) {
            for (int i{ 0 }; i < TM; ++i) {
                regA[i] = As[dotIdx * BM + threadRow * TM + i];
            }

            for (int i{ 0 }; i < TN; ++i) {
                regB[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
            }

            for (int resM{ 0 }; resM < TM; ++resM) {
                for (int resN{ 0 }; resN < TN; ++resN) {
                    resultTh[resM * TN + resN] += regA[resM] * regB[resN];
                }
            }
        }
        __syncthreads();
    }

    for (int resM{ 0 }; resM < TM; ++resM) {
        for (int resN{ 0 }; resN < TN; resN += 4) {
            float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0];
            tmp.x = alpha * resultTh[resM * TN + resN + 0] + beta * tmp.x;
            tmp.y = alpha * resultTh[resM * TN + resN + 1] + beta * tmp.y;
            tmp.z = alpha * resultTh[resM * TN + resN + 2] + beta * tmp.z;
            tmp.w = alpha * resultTh[resM * TN + resN + 3] + beta * tmp.w;
            reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0] = tmp;
        }
    }
}

void launch_smem_resolve_bankconflicts_1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 blockDim{ (BM * BN) / (TM * TN), 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_resolve_bankconflicts_1<BM, BN, BK, TM, TN> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}


template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_smem_resolve_bankconflicts_2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    const int extraCols = 5;
    __shared__ float Bs[BK * (BN + extraCols)];
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint colAs = threadIdx.x % (BK / 4);
    const uint rowAs = threadIdx.x / (BK / 4);
    const uint colBs = threadIdx.x % (BN / 4);
    const uint rowBs = threadIdx.x / (BN / 4);

    float resultTh[TM * TN] = { 0.0f };
    float regA[TM] = { 0.0f };
    float regB[TN] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        float4 tmp = reinterpret_cast<const float4*>(&A[rowAs * K + colAs * 4])[0];
        As[(colAs * 4 + 0) * BM + rowAs] = tmp.x;
        As[(colAs * 4 + 1) * BM + rowAs] = tmp.y;
        As[(colAs * 4 + 2) * BM + rowAs] = tmp.z;
        As[(colAs * 4 + 3) * BM + rowAs] = tmp.w;

        tmp = reinterpret_cast<const float4*>(&B[rowBs * N + colBs * 4])[0];
        Bs[rowBs * (BN + extraCols) + colBs * 4 + 0] = tmp.x;
        Bs[rowBs * (BN + extraCols) + colBs * 4 + 1] = tmp.y;
        Bs[rowBs * (BN + extraCols) + colBs * 4 + 2] = tmp.z;
        Bs[rowBs * (BN + extraCols) + colBs * 4 + 3] = tmp.w;
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{ 0 }; dotIdx < BK; ++dotIdx) {
            for (int i{ 0 }; i < TM; ++i) {
                regA[i] = As[dotIdx * BM + threadRow * TM + i];
            }

            for (int i{ 0 }; i < TN; ++i) {
                regB[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
            }

            for (int resM{ 0 }; resM < TM; ++resM) {
                for (int resN{ 0 }; resN < TN; ++resN) {
                    resultTh[resM * TN + resN] += regA[resM] * regB[resN];
                }
            }
        }
        __syncthreads();
    }

    for (int resM{ 0 }; resM < TM; ++resM) {
        for (int resN{ 0 }; resN < TN; resN += 4) {
            float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0];
            tmp.x = alpha * resultTh[resM * TN + resN + 0] + beta * tmp.x;
            tmp.y = alpha * resultTh[resM * TN + resN + 1] + beta * tmp.y;
            tmp.z = alpha * resultTh[resM * TN + resN + 2] + beta * tmp.z;
            tmp.w = alpha * resultTh[resM * TN + resN + 3] + beta * tmp.w;
            reinterpret_cast<float4*>(&C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0] = tmp;
        }
    }
}

void launch_smem_resolve_bankconflicts_2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 blockDim{ (BM * BN) / (TM * TN), 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_resolve_bankconflicts_2<BM, BN, BK, TM, TN> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}

static constexpr uint WARPSIZE = 32;
template <const uint BM, const uint BN, const uint BK, const uint WM, const uint WN,
          const uint WNITER, const uint TM, const uint TN, const uint THREAD_NUMS>
__global__ void __launch_bounds__(THREAD_NUMS) gemm_smem_warptiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    // Move C_ptr to warp's output tile
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint rowAs = threadIdx.x / (BK / 4);
    const uint colAs = threadIdx.x % (BK / 4);
    const uint rowBs = threadIdx.x / (BN / 4);
    const uint colBs = threadIdx.x % (BN / 4);
    constexpr uint rowStrideA = (THREAD_NUMS * 4) / BK;
    constexpr uint rowStrideB = THREAD_NUMS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float resultsTh[WMITER * TM * WNITER * TN] = { 0.0f };
    // we cache into registers on the warptile level
    float regA[WMITER * TM] = { 0.0f };
    float regB[WNITER * TN] = { 0.0f };
    for (int bkIdx{ 0 }; bkIdx < K; bkIdx += BK) {
        for (uint offset{ 0 }; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(&A[(rowAs + offset) * K + colAs * 4])[0];
            As[(colAs * 4 + 0) * BM + rowAs + offset] = tmp.x;
            As[(colAs * 4 + 1) * BM + rowAs + offset] = tmp.y;
            As[(colAs * 4 + 2) * BM + rowAs + offset] = tmp.z;
            As[(colAs * 4 + 3) * BM + rowAs + offset] = tmp.w;
        }

        for (uint offset{ 0 }; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(&Bs[(rowBs + offset) * BN + colBs * 4])[0] =
                reinterpret_cast<const float4*>(&B[(rowBs + offset) * N + colBs * 4])[0];
        }
        __syncthreads();

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // populate registers for whole warptile
            for (uint wSubRowIdx{ 0 }; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i{ 0 }; i < TM; ++i) {
                    regA[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx{ 0 }; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i{ 0 }; i < TN; ++i) {
                    regB[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
                }
            }

            // execute warptile matmul
            for (uint wSubRowIdx{ 0 }; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx{ 0 }; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (uint resIdxM{ 0 }; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN{ 0 }; resIdxN < TN; ++resIdxN) {
                            resultsTh[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] +=
                                regA[wSubRowIdx * TM + resIdxM] * regB[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    for (uint wSubRowIdx{ 0 }; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx{ 0 }; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM{ 0 }; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN{ 0 }; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * resultsTh[i + 0] + beta * tmp.x;
                    tmp.y = alpha * resultsTh[i + 1] + beta * tmp.y;
                    tmp.z = alpha * resultsTh[i + 2] + beta * tmp.z;
                    tmp.w = alpha * resultsTh[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

void launch_smem_warptiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const uint BM = 64;
    const uint BN = 128;
    const uint BK = 16;
    const uint WM = 32;
    const uint WN = 64;
    const uint WNITER = 2;
    const uint TM = 4;
    const uint TN = 4;
    const uint THREAD_NUMS = 128;

    dim3 blockDim{ THREAD_NUMS, 1, 1 };
    dim3 gridDim{ CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1 };

    gemm_smem_warptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, THREAD_NUMS> <<< gridDim, blockDim >>> (M, N, K, alpha, A, B, beta, C);
}

void launch_culas_float(cublasHandle_t handle, int M, int N, int K, float alpha, const float* A, float* B, float beta, float* C) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, B, CUDA_R_32F, N,
                A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, 
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void preprocess(int dev, bool promptName/* = false */) {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));

    if (promptName)
        printf("GPU name: %s\n", prop.name);
}

void postprocess() {
    cudaDeviceReset();
}

const char* KTStr(KType emKernelType) {
    const char* res{ nullptr };
    switch (emKernelType) {
        case KType::KT_NAIVE:                         res = "naive";                         break;
        case KType::KT_GMEM_COALESCING:               res = "gmem coalescing";               break;
        case KType::KT_SMEM_CACHING:                  res = "smem caching";                  break;
        case KType::KT_SMEM_1D_BLOCKTILING:           res = "smem 1D blocking";              break;
        case KType::KT_SMEM_2D_BLOCKTILING:           res = "smem 2D blocking";              break;
        case KType::KT_SMEM_VECTORIZE:                res = "smem vectorize";                break;
        case KType::KT_SMEM_RESOLVE_BANKCONFLICTS_1:  res = "smem resolve bank conflicts 1"; break;
        case KType::KT_SMEM_RESOLVE_BANKCONFLICTS_2:  res = "smem resolve bank conflicts 2"; break;
        case KType::KT_SMEM_WARPTILING:               res = "smem warp tiling";              break;
        case KType::KT_CUBLAS:                        res = "cublasGemmEx";                  break;
        default: break;
    }
    return res;
}

double runKernel(KType emKernelType, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, int repeatTimes/* = 1*/) {
    const unsigned bytesA = M * K * sizeof(float);
    const unsigned bytesB = K * N * sizeof(float);
    const unsigned bytesC = M * N * sizeof(float);
    
    // allocate device side resources
    float* Ad = nullptr;
    float* Bd = nullptr;
    float* Cd = nullptr;
    CHECK(cudaMalloc(&Ad, bytesA));
    CHECK(cudaMalloc(&Bd, bytesB));
    CHECK(cudaMemcpy(Ad, A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Bd, B, bytesB, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&Cd, bytesC));
    CHECK(cudaMemset(Cd, 0, bytesC));

    // run kernel and statistics elapsed time
    cublasHandle_t handle;
    if (emKernelType == KType::KT_CUBLAS) {
        if (cublasCreate(&handle)) {
            printf("Create cublas handle error.\n");
            exit(EXIT_FAILURE);
        }
        // cuBLAS wramup
        launch_culas_float(handle, M, N, K, alpha, Ad, Bd, beta, Cd);
        CHECK(cudaDeviceSynchronize());
    }
    float elapsedTime{ 0.0f };
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int rt{ 0 }; rt < repeatTimes; ++rt) {
        switch (emKernelType) {
            case KType::KT_NAIVE:
                launch_naive(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_GMEM_COALESCING:
                launch_gmem_coalescing(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_CACHING:
                launch_smem_caching(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_1D_BLOCKTILING:
                launch_smem_1D_blocktiling(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_2D_BLOCKTILING:
                launch_smem_2D_blocktiling(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_VECTORIZE:
                launch_smem_vectorize(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_RESOLVE_BANKCONFLICTS_1:
                launch_smem_resolve_bankconflicts_1(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_RESOLVE_BANKCONFLICTS_2:
                launch_smem_resolve_bankconflicts_2(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_SMEM_WARPTILING:
                launch_smem_warptiling(M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            case KType::KT_CUBLAS:
                launch_culas_float(handle, M, N, K, alpha, Ad, Bd, beta, Cd);
                break;

            default:
                break;
        }
    }
    cudaEventRecord(end);
    CHECK(cudaEventSynchronize(end));
    cudaEventElapsedTime(&elapsedTime, beg, end);
    elapsedTime /= 1000.; // Convert to seconds

    double flops = 2 * int64_t(M) * int64_t(N) * int64_t(K);
    printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: (%d). kernel: (%s) \n",
        elapsedTime / repeatTimes, repeatTimes * flops * 1e-9 / elapsedTime, M, KTStr(emKernelType));
    
    // return result
    CHECK(cudaMemcpy(C, Cd, bytesC, cudaMemcpyDeviceToHost));

    // free resources
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(&Ad);
    cudaFree(&Bd);
    cudaFree(&Cd);

    // return elapsed time
    return elapsedTime;
}