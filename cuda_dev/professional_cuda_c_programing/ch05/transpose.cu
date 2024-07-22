#include "../common/common.h"
#include <cstdlib>


static constexpr int BDIMX = 32;
static constexpr int BDIMY = 32;


__global__ void transposeGmem_naive(float* out, float* in, int nx, int ny) {
    uint ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void copyGmem(float* out, float* in, int nx, int ny) {
    uint ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

static void randomInit(float* data, int size) {
    for (int i{ 0 }; i < size; ++i) {
        data[i] = static_cast<float>(rand() / static_cast<float>(RAND_MAX));
    }
}


int main() {
    SelectGPUDevice(0);

    const int NX = 4096;
    const int NY = 4096;
    const int size = NX * NY;
    const int bytes = size * sizeof(float);
    float* h_idata = (float*)malloc(bytes);
    randomInit(h_idata, size);

    float* d_idata = nullptr;
    float* d_odata = nullptr;
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, bytes);
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    dim3 bDim{BDIMX, BDIMY};
    dim3 gDim{(NX + bDim.x - 1) / bDim.x, (NY + bDim.y - 1) / bDim.y};


    cudaMemset(d_odata, 0, bytes);
    double start = seconds();
    transposeGmem_naive <<< gDim, bDim >>> (d_odata, d_idata, NX, NY);
    CHECK(cudaDeviceSynchronize());
    double elaps = seconds() - start;
    printf("transposeGmem_naive <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", gDim.x, gDim.y, bDim.x, bDim.y, elaps);


    cudaMemset(d_odata, 0, bytes);
    start = seconds();
    copyGmem <<< gDim, bDim >>> (d_odata, d_idata, NX, NY);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("copyGmem <<< (%d, %d), (%d, %d) >>> elaps %f sec\n", gDim.x, gDim.y, bDim.x, bDim.y, elaps);
    

    cudaFree(&d_idata);
    cudaFree(&d_odata);
    free(h_idata);
    return 0;
}
