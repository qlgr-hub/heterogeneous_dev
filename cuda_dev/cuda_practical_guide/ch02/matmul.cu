#include "../common/utils.h"
#include "../common/common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>


// static void hostmult0(float* C, float* A, float* B, uint32_t Ay, uint32_t Ax, uint32_t Bx) {
static void hostmult0(float* __restrict C, const float* __restrict A, const float* __restrict B,
    uint32_t Ay, uint32_t Ax, uint32_t Bx) {
    for (uint32_t i = 0; i < Ay; ++i) {
        for (uint32_t j = 0; j < Bx; ++j) {
            C[i * Bx + j] = 0.f;
            for (uint32_t k = 0; k < Ax; ++k) {
                C[i * Bx + j] += A[i * Ax + k] * B[k * Bx + j];
            }
        }
    }
}


// __global__ void gpumult0(float* C, const float* A, const float* B, uint32_t Ay, uint32_t Ax, uint32_t Bx) {
__global__ void gpumult0(float* __restrict C, const float* __restrict A, const float* __restrict B, uint32_t Ay, uint32_t Ax, uint32_t Bx) {
    uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty >= Ay || tx >= Bx)
        return;

    auto idx = [&Bx](uint32_t i, uint32_t j) {
        return i * Bx + j;
    };

    C[idx(ty, tx)] = 0.f;
    for (uint32_t k = 0; k < Ax; ++k) {
        // C[ty * Bx + tx] += A[ty * Bx + k] * B[k * Bx + tx];
        C[idx(ty, tx)] += A[idx(ty, k)] * B[idx(k, tx)];
    }
}


template <uint32_t TS>
__global__ void gputiled(float* __restrict C, const float* __restrict A, const float* __restrict B, uint32_t Ay, uint32_t Ax, uint32_t Bx) {
    __shared__ float Atile[TS][TS];
    __shared__ float Btile[TS][TS];

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t ocx = blockDim.x * blockIdx.x;
    uint32_t ocy = blockDim.y * blockIdx.y;

    uint32_t ax = tx;
    uint32_t ay = ocy + ty;
    uint32_t bx = ocx + tx;
    uint32_t by = ty;

    float csum = 0.f;
#pragma unroll 16
    for (uint32_t t = 0; t < gridDim.x; ++t) {
        Atile[ty][tx] = A[ay * Ax + ax];
        Btile[ty][tx] = B[by * Bx + bx];
        __syncthreads();

        for (uint32_t k = 0; k < TS; ++k) {
            csum += Atile[ty][k] * Btile[k][tx];
        }
        __syncthreads();

        ax += TS;
        by += TS;
    }
    C[ay * Bx + bx] = csum;
}


int main(int argc, char* argv[]) {
    uint32_t Arow = (argc > 1) ? atoi(argv[1]) : 1024;
    uint32_t Acol = (argc > 2) ? atoi(argv[2]) : Arow;
    uint32_t Brow = Acol;
    uint32_t Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
    uint32_t Crow = Arow;
    uint32_t Ccol = Bcol;

    thrust::host_vector<float> A(Arow * Acol);
    thrust::host_vector<float> B(Brow * Bcol);
    thrust::host_vector<float> C(Crow * Ccol);

    std::default_random_engine gen{ 12345678 };
    std::uniform_real_distribution<float> fran{ 0.f, 1.f };
    
    for (uint32_t k = 0; k < Arow * Acol; ++k) {
        A[k] = fran(gen);
    }

    for (uint32_t k = 0; k < Brow * Bcol; ++k) {
        B[k] = fran(gen);
    }

    Utils::TimeStats tim;
    hostmult0(C.data(), A.data(), B.data(), Acol, Arow, Brow);
    double t1 = tim.lap_milli();

    double flops = 2.0 * double(Arow * Acol * Bcol);
    double gflops = flops / (t1 * 1000000.0);
    double gbytes = gflops * 6.0; // 12 bytes per ter
    printf("A %d x %d B %d x %d host time %.3f ms Gflops/sec %.3f Gbytes %.3f\n",
        Arow, Acol, Brow, Bcol, t1, gflops, gbytes);


    uint32_t tilex = (argc > 4) ? atoi(argv[4]) : 32;
    uint32_t tiley = (argc > 5) ? atoi(argv[5]) : 8;

    thrust::device_vector<float> dev_A(Arow * Acol);
    thrust::device_vector<float> dev_B(Brow * Bcol);
    thrust::device_vector<float> dev_C(Crow * Ccol);
    thrust::device_vector<float> dev_D(Crow * Ccol);
    dev_A = A; // H2D copy
    dev_B = B; // H2D copy

    dim3 threads{ tilex, tiley, 1 };
    dim3 blocks{ (Bcol + threads.x - 1) / threads.x, (Arow + threads.y - 1) / threads.y, 1 };

    tim.reset();
    gpumult0 <<< blocks, threads >>> (dev_C.data().get(),
        dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    CHECK(cudaDeviceSynchronize());
    double t2 = tim.lap_milli();

    thrust::host_vector<float> C1(Crow * Ccol);
    C1 = dev_C; // D2H copy

    // check result
    // for (uint32_t i = 0; i < Crow * Ccol; ++i) {
    //     if ((C[i] - C1[i]) >= 1e-3) {
    //         printf("not match: %d, %7.6f, %7.6f\n", i, C[i], C1[i]);
    //         break;
    //     }
    // }

    gflops = flops / (t2 * 1000000.0);
    gbytes = gflops * 6.0; // 12 bytes per ter
    printf("A %d x %d B %d x %d gpu time %.3f ms Gflops/sec %.3f Gbytes %.3f\n",
        Arow, Acol, Brow, Bcol, t2, gflops, gbytes);


    dev_A = A; // H2D copy
    dev_B = B; // H2D copy
    tim.reset();
    if (tilex == 8) {
        gputiled<8> <<<blocks,threads>>> (dev_C.data().get(),
            dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    }
	else if (tilex == 16) {
        gputiled<16> <<<blocks,threads>>> (dev_C.data().get(),
            dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    }
	else if (tilex == 32) {
        gputiled<32> <<<blocks,threads>>> (dev_C.data().get(),
            dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    }
    CHECK(cudaDeviceSynchronize());
    double t3 = tim.lap_milli();
    C1 = dev_C; // D2H copy

    // check result
    // for (uint32_t i = 0; i < Crow * Ccol; ++i) {
    //     if ((C[i] - C1[i]) >= 1e-3) {
    //         printf("not match: %d, %7.6f, %7.6f\n", i, C[i], C1[i]);
    //         break;
    //     }
    // }

    gflops = flops / (t3 * 1000000.0);
    gbytes = gflops * 6.0; // 12 bytes per ter
    printf("A %d x %d B %d x %d gputiled time %.3f ms Gflops/sec %.3f Gbytes %.3f\n",
        Arow, Acol, Brow, Bcol, t3, gflops, gbytes);


    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    dev_A = A; // H2D copy
    dev_B = B; // H2D copy
    float alpha = 1.f;
    float beta  = 1.f;

    // cublas need warmup
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, Crow, Ccol, Arow, &alpha,
        dev_A.data().get(), Acol, dev_B.data().get(), Bcol, &beta, dev_C.data().get(), Ccol);

    tim.reset();
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, Crow, Ccol, Arow, &alpha,
        dev_A.data().get(), Acol, dev_B.data().get(), Bcol, &beta, dev_C.data().get(), Ccol);
    beta  = 0.f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, Crow, Ccol, &alpha,
        dev_C.data().get(), Crow, &beta, dev_C.data().get(), Crow, dev_D.data().get(), Ccol);
    CHECK(cudaDeviceSynchronize());
    double t4 = tim.lap_milli();

    C1 = dev_D; // D2H copy

    // check result
    for (uint32_t i = 0; i < Crow * Ccol; ++i) {
        if ((C[i] - C1[i]) >= 1e-3) {
            printf("not match: %d, %7.6f, %7.6f\n", i, C[i], C1[i]);
            break;
        }
    }
    
    gflops = flops / (t4 * 1000000.0);
    gbytes = gflops * 6.0; // 12 bytes per ter
    printf("A %d x %d B %d x %d cublasSgemm time %.3f ms Gflops/sec %.3f Gbytes %.3f\n",
        Arow, Acol, Brow, Bcol, t4, gflops, gbytes);

    return 0;
}
