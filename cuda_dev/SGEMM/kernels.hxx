#pragma once

enum class KType {
    KT_NAIVE,
    KT_GMEM_COALESCING,
    KT_SMEM_CACHING,
    KT_SMEM_1D_BLOCKTILING,
    KT_SMEM_2D_BLOCKTILING,
    KT_SMEM_VECTORIZE,
    KT_SMEM_RESOLVE_BANKCONFLICTS_1,
    KT_SMEM_RESOLVE_BANKCONFLICTS_2,
    KT_SMEM_WARPTILING,
    KT_CUBLAS
};

void   preprocess(int dev, bool promptName = false);
void   postprocess();
double runKernel(KType emKernelType, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, int repeatTimes = 1);
