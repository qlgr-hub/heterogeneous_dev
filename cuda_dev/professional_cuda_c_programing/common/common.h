#pragma once

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include <cstdio>

// instead of macro
static inline void CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
}

// instead of macro
static inline int CEIL_DIV(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

// instead of macro
static inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
};

void SelectGPUDevice(int dev, bool promptName = true);

