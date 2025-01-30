#pragma once

#include "Complex.cuh"
#include <cuda_runtime.h>


__device__ float mandlebrot_iterate(Complex C);
__global__ void mandelbrot_kernel(int* imagePtr, int width, int height);