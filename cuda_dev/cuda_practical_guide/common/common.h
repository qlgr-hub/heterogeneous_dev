#pragma once

#include <cuda_runtime.h>
#include <cstdio>


static inline void _cuda_rt_check(cudaError_t err, const char* filename, int lineNo) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s:%d, ", filename, lineNo);
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
}
#define CHECK(err) _cuda_rt_check(err, __FILE__, __LINE__)



