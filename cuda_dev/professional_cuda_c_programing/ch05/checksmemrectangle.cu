#include "../common/common.h"


static constexpr int BDIMX = 32;
static constexpr int BDIMY = 16;
static constexpr int IPAD  = 1;

__global__ void setRowReadRow(int* out) {
    __shared__ int tile[BDIMY][BDIMX];
    
    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out) {
    __shared__ int tile[BDIMX][BDIMY];
    
    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int* out) {
    __shared__ int tile[BDIMY][BDIMX];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    uint irow = idx / blockDim.y;
    uint icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColPad(int* out) {
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    uint irow = idx / blockDim.y;
    uint icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}


int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("start, device: %s\n", prop.name);
    CHECK(cudaSetDevice(dev));

    
    int* result1 = nullptr;
    cudaMalloc(&result1, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result1, 0, BDIMX * BDIMY * sizeof(int));

    dim3 gridDim{1, 1};
    dim3 blockDim{BDIMX, BDIMY};
    double start = seconds();
    setRowReadRow <<< gridDim, blockDim >>> (result1);
    CHECK(cudaDeviceSynchronize());
    double elaps = seconds() - start;
    printf("setRowReadRow <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, elaps);


    int* result2 = nullptr;
    cudaMalloc(&result2, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result2, 0, BDIMX * BDIMY * sizeof(int));

    start = seconds();
    setColReadCol <<< gridDim, blockDim >>> (result2);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("setColReadCol <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, elaps);


    int* result3 = nullptr;
    cudaMalloc(&result3, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result3, 0, BDIMX * BDIMY * sizeof(int));

    start = seconds();
    setRowReadCol <<< gridDim, blockDim >>> (result3);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("setRowReadCol <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, elaps);


    int* result4 = nullptr;
    cudaMalloc(&result4, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result4, 0, BDIMX * BDIMY * sizeof(int));

    start = seconds();
    setRowReadColPad <<< gridDim, blockDim >>> (result4);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;
    printf("setRowReadColPad <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, elaps);


    cudaFree(&result1);
    cudaFree(&result2);
    cudaFree(&result3);
    cudaFree(&result4);
    return 0;
}