#include "../common/common.h"

static constexpr int BDIMX = 32;
static constexpr int BDIMY = 32;
static constexpr int IPAD = 1;

__global__ void setRowReadRow(int* out) {
    __shared__ int tile[BDIMX][BDIMY];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int* out) {
    __shared__ int tile[BDIMX][BDIMY];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol(int* out) {
    __shared__ int tile[BDIMX][BDIMY];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int* out) {
    extern __shared__ int tile[];

    uint row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    uint col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_idx] = row_idx;

    __syncthreads();

    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int* out) {
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    uint idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int* out) {
    extern __shared__ int tile[];

    uint row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    uint col_idx = threadIdx.x * (blockDim.y + IPAD) + threadIdx.y;
    uint g_idx = threadIdx.y * blockDim.x+ threadIdx.x;

    tile[row_idx] = g_idx;

    __syncthreads();

    out[g_idx] = tile[col_idx];
}

int main() {

    // select GPU device
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Using Device %d: %s\n", dev, prop.name);
    CHECK(cudaSetDevice(dev));

    // check shared memory config
    // cudaSharedMemConfig cfg;
    // CHECK(cudaDeviceGetSharedMemConfig(&cfg));
    // printf("cudaSharedMemConfig: %d\n", cfg);

    // check cache config
    cudaFuncCache cfg;
    CHECK(cudaDeviceGetCacheConfig(&cfg));
    printf("cudaFuncCache: %d\n", cfg);

    int* result = nullptr;
    cudaMalloc((int**)&result, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result, 0, BDIMX * BDIMY * sizeof(int));

    // 
    dim3 gridDim{ 1, 1 };
    dim3 blockDim{ BDIMX, BDIMY };
    double dStart = seconds();
    setRowReadCol <<< gridDim, blockDim >>> (result);
    CHECK(cudaDeviceSynchronize());
    double dElaps = seconds() - dStart;
    printf("setRowReadCol <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    int* result2 = nullptr;
    cudaMalloc((int**)&result2, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result2, 0, BDIMX * BDIMY * sizeof(int));

    dStart = seconds();
    setRowReadRow <<< gridDim, blockDim >>> (result2);
    CHECK(cudaDeviceSynchronize());
    dElaps = seconds() - dStart;
    printf("setRowReadRow <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    int* result3 = nullptr;
    cudaMalloc((int**)&result3, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result3, 0, BDIMX * BDIMY * sizeof(int));

    dStart = seconds();
    setColReadCol <<< gridDim, blockDim >>> (result3);
    CHECK(cudaDeviceSynchronize());
    dElaps = seconds() - dStart;
    printf("setColReadCol <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    int* result4 = nullptr;
    cudaMalloc((int**)&result4, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result4, 0, BDIMX * BDIMY * sizeof(int));

    dStart = seconds();
    setRowReadColDyn <<< gridDim, blockDim, BDIMX * BDIMY * sizeof(int) >>> (result4);
    CHECK(cudaDeviceSynchronize());
    dElaps = seconds() - dStart;
    printf("setRowReadColDyn <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    int* result5 = nullptr;
    cudaMalloc((int**)&result5, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result5, 0, BDIMX * BDIMY * sizeof(int));

    dStart = seconds();
    setRowReadColPad <<< gridDim, blockDim>>> (result5);
    CHECK(cudaDeviceSynchronize());
    dElaps = seconds() - dStart;
    printf("setRowReadColPad <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    int* result6 = nullptr;
    cudaMalloc((int**)&result6, BDIMX * BDIMY * sizeof(int));
    cudaMemset(result6, 0, BDIMX * BDIMY * sizeof(int));

    dStart = seconds();
    setRowReadColDynPad <<< gridDim, blockDim, (BDIMX + IPAD) * BDIMY * sizeof(int) >>> (result6);
    CHECK(cudaDeviceSynchronize());
    dElaps = seconds() - dStart;
    printf("setRowReadColDynPad <<< (%d, %d), (%d, %d) >>> Time elapsed %f sec\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, dElaps);


    cudaFree(&result);
    cudaFree(&result2);
    cudaFree(&result3);
    cudaFree(&result4);
    cudaFree(&result5);
    cudaFree(&result6);
    return 0;
}