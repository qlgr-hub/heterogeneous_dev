#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <thrust/device_vector.h>

#include "../common/utils.h"


__host__ __device__ inline float sinsum(float x, uint32_t terms) {
    // sin(x) = x - x^3/3! + x^5/5! ...
    float term = x;  // first term of series
    float sum = term; // sum of terms so far
    float x2 = x * x;
    for (uint32_t n = 1; n < terms; ++n) {
        term *= -x2 / float(2 * n * (2 * n + 1));
        sum += term;
    }
    return sum;
}

__global__ void gpu_sin(float* sums, uint32_t steps, uint32_t terms, float step_size) {
    // unique thread ID
    uint32_t step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step < steps) {
        float x = step_size * step;
        sums[step] = sinsum(x, terms); // store sums
    }
}


int main(int argc, char * argv[]) {
    uint32_t steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    uint32_t terms = (argc > 2) ? atoi(argv[2]) : 1000;
    uint32_t threads = (argc > 3) ? atoi(argv[3]) : 4;
    uint32_t gpu_threads = 1024;
    int32_t gpu_blocks = (steps + gpu_threads - 1) / gpu_threads;

    double pi = 3.14159265358979323;
    double step_size = pi / (steps - 1); // n-1 steps

    Utils::TimeStats tim;
    double cup_sum = 0.0;
    for (uint32_t step = 0; step < steps; ++step) {
        float x = step_size * step;
        cup_sum += sinsum(x, terms);  // sum of Taylor series
    }
    double cpu_time = tim.lap_milli();  // elapsed time

    // Trapezoidal Rule correction
    cup_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
    cup_sum *= step_size;
    printf("cpu sum = %.10f, steps %d, terms %d time %.3f ms\n",
        cup_sum, steps, terms, cpu_time);

    tim.reset();
    double omp_sum = 0.0;
    omp_set_num_threads(threads);
#pragma omp parallel for reduction (+:omp_sum)
    for (uint32_t step = 0; step < steps; ++step) {
        float x = step_size * step;
        omp_sum += sinsum(x, terms);  // sum of Taylor series
    }
    double omp_time = tim.lap_milli();  // elapsed time

    // Trapezoidal Rule correction
    omp_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
    omp_sum *= step_size;
    printf("omp sum = %.10f, steps %d, terms %d time %.3f ms (%d threads)\n",
        omp_sum, steps, terms, omp_time, threads);

    thrust::device_vector<float> dsums(steps);
    float* dptr = thrust::raw_pointer_cast(&dsums[0]);
    tim.reset();
    gpu_sin <<<gpu_blocks, gpu_threads>>> (dptr, steps, terms, step_size);
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
    double gpu_time = tim.lap_milli();  // elapsed time

    // Trapezoidal Rule correction
    gpu_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    printf("gpu sum = %.10f, steps %d, terms %d time %.3f ms\n",
        gpu_sum, steps, terms, gpu_time);
    
    return 0;
}