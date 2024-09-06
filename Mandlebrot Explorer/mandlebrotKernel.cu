#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>

#include "complex.cu"


__device__ float mandlebrot_iterate(Complex C)
{
    int max_iterations = 50;
    Complex Zn = Complex(0, 0);
    int iterations_countdown = max_iterations;

    while (--iterations_countdown)
    {
        if (Zn.magnitude() >= 2.0) { return max_iterations - iterations_countdown; }
        Zn = Zn * Zn + C;
    }
    return 0;
}

__global__ void mandelbrot_kernel(int* imagePtr, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Complex C((x - width / 2.0f) / (width / 4.0f), (y - height / 2.0f) / (height / 4.0f) - 0.5f);
    imagePtr[y * width + x] = mandlebrot_iterate(C);
}