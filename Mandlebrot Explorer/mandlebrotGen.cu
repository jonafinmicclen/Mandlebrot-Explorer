#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>

#include "complex.cu"
#include "mandlebrotKernel.cu"

int* generateImage(int arr_width, int arr_height, int scale, int center_x, int center_y) {

    const int size_of_arr = arr_height * arr_width;
    int size_of_arr_bytes = sizeof(int) * size_of_arr;

    int* imagePtr = new int[size_of_arr];
    int* imagePtr_CUDA;

    cudaMalloc((void**)&imagePtr_CUDA, size_of_arr_bytes);

    dim3 threadsPerBlock(10, 10);
    dim3 blocksPerGrid((arr_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (arr_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandelbrot_kernel << <blocksPerGrid, threadsPerBlock >> > (imagePtr_CUDA, arr_width, arr_height);

    cudaMemcpy(imagePtr, imagePtr_CUDA, size_of_arr_bytes, cudaMemcpyDeviceToHost);
    cudaFree(imagePtr_CUDA);

    return imagePtr;
}
