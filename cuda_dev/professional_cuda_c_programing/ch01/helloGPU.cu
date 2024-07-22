#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
   printf("Hello from GPU!\n");
}

int main() {
    std::cout << "Hello from CPU!\n";

    helloFromGPU <<<1, 10>>> ();
    
    cudaDeviceReset();
    return 0;
}