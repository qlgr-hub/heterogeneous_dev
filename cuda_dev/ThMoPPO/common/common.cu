#include "common.h"

void SelectGPUDevice(int dev, bool promptName/* = true */) {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));

    if (promptName)
        printf("GPU name: %s\n", prop.name);
}

