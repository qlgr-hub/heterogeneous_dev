#include "../common/common.h"
#include <cstdlib>
#include <ctime>


static constexpr int DIM = 256;

static void randomInit(int* data, int size) {
    for (int i{ 0 }; i < size; ++i) {
        data[i] = static_cast<int>(rand() & 0xFF);
    }
}

static int recursiveReduce(int* data, int size) {
    if (size == 1) return data[0];

    const int stride = size / 2;
    for (int i{ 0 }; i < stride; ++i) {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

__global__ void redudeGmem(int* g_idata, int* g_odata, uint n) {
    uint tid = threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x;

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = idata;

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void redudeSmem(int* g_idata, int* g_odata, uint n) {
    __shared__ int smem[DIM];

    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int* idata = g_idata + blockIdx.x * blockDim.x;

    smem[tid] = idata[tid];
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = smem;

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void redudeGmemUnroll(int* g_idata, int* g_odata, uint n) {
    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;
    
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
     __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = idata;

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void redudeSmemUnroll(int* g_idata, int* g_odata, uint n) {
    __shared__ int smem[DIM];

    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    int tempSum = 0;
    if (idx + 3 * blockDim.x <= n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tempSum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tempSum;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = smem;

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int main() {
    srand(time(nullptr));

    int size = 1 << 24;
    int bytes = size * sizeof(int);
    int* temp = (int*)malloc(bytes);
    randomInit(temp, size);
    
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));
    // printf("use device: %s\n", prop.name);

    dim3 bDim{DIM, 1};
    dim3 gDim{(size + bDim.x - 1) / bDim.x, 1};
    printf("grid.x: %d, block.x: %d\n", gDim.x, bDim.x);

    int* h_odata = (int*)malloc(gDim.x * sizeof(int));
    int* h_idata = (int*)malloc(bytes);
    memcpy(h_idata, temp, bytes);

    double bgn = seconds();
    int cpuRes = recursiveReduce(temp, size);
    double elaps = seconds() - bgn;
    printf("CPU result: %d, elaps: %f sec\n", cpuRes, elaps);

    int* d_idata = nullptr;
    int* d_odata = nullptr;
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, gDim.x * sizeof(int));

    cudaMemset(d_odata, 0, gDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    bgn = seconds();
    redudeGmem <<< gDim, bDim >>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - bgn;
    cudaMemcpy(h_odata, d_odata, gDim.x * sizeof(int), cudaMemcpyDeviceToHost);

    bgn = seconds();
    int gpuRes = 0;
    for (int i{ 0 }; i < gDim.x; ++i) {
        gpuRes += h_odata[i];
    }
    elaps += seconds() - bgn;
    printf("GPU result(GMEM): %d, elaps: %f sec\n", gpuRes, elaps);


    cudaMemset(d_odata, 0, gDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    bgn = seconds();
    redudeSmem <<< gDim, bDim >>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - bgn;
    cudaMemcpy(h_odata, d_odata, gDim.x * sizeof(int), cudaMemcpyDeviceToHost);

    bgn = seconds();
    int gpuRes_1 = 0;
    for (int i{ 0 }; i < gDim.x; ++i) {
        gpuRes_1 += h_odata[i];
    }
    elaps += seconds() - bgn;
    printf("GPU result(SMEM): %d, elaps: %f sec\n", gpuRes_1, elaps);


    cudaMemset(d_odata, 0, gDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    bgn = seconds();
    redudeGmemUnroll <<< gDim.x / 4, bDim >>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - bgn;
    cudaMemcpy(h_odata, d_odata, gDim.x / 4 * sizeof(int), cudaMemcpyDeviceToHost);

    bgn = seconds();
    int gpuRes_2 = 0;
    for (int i{ 0 }; i < gDim.x / 4; ++i) {
        gpuRes_2 += h_odata[i];
    }
    elaps += seconds() - bgn;
    printf("GPU result(GMEM Unroll): %d, elaps: %f sec\n", gpuRes_2, elaps);


    cudaMemset(d_odata, 0, gDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    bgn = seconds();
    redudeSmemUnroll <<< gDim.x / 4, bDim >>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - bgn;
    cudaMemcpy(h_odata, d_odata, gDim.x / 4 * sizeof(int), cudaMemcpyDeviceToHost);

    bgn = seconds();
    int gpuRes_3 = 0;
    for (int i{ 0 }; i < gDim.x / 4; ++i) {
        gpuRes_3 += h_odata[i];
    }
    elaps += seconds() - bgn;
    printf("GPU result(SMEM Unroll): %d, elaps: %f sec\n", gpuRes_3, elaps);


    cudaFree(&d_idata);
    cudaFree(&d_odata);
    free(temp);
    free(h_idata);
    free(h_odata);
    return 0;
}