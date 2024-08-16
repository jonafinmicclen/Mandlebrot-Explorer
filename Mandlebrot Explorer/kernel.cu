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

int main() {
    const int arr_width = 100;
    const int arr_height = 100;
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

    for (int y = 0; y < arr_height; ++y) {
        for (int x = 0; x < arr_width; ++x) {
            std::cout << imagePtr[y * arr_width + x] << " ";
        }
        std::cout << "\n";
    }

    delete[] imagePtr;

    return 0;
}
