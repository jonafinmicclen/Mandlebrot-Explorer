#pragma once

#include "Complex.cuh"
#include <cuda_runtime.h>
#include "Constants.h"

#include <stdio.h>


__device__ void mandlebrot_iterate(Complex C, int* imagePtr);
__global__ void mandelbrot_kernel(int* imagePtr, int width, int height,
    float x_zoom, float y_zoom, float x_offset, float y_offset);
