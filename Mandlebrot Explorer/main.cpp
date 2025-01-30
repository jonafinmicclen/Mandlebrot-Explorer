#include "Mandlebrot.cuh"
#include <iostream>
#include "device_launch_parameters.h"

// Image array parameters
const int arr_width = 100;
const int arr_height = 100;
const int size_of_arr = arr_height * arr_width;
int size_of_arr_bytes = sizeof(int) * size_of_arr;

// Allocate device and host memory
int* imagePtr = new int[size_of_arr];
int* imagePtr_CUDA;

dim3 threadsPerBlock(10, 10);
dim3 blocksPerGrid((arr_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (arr_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

void cleanupMemory() {
    // Cleanup memory
    cudaFree(imagePtr_CUDA);
    delete[] imagePtr;
}

void allocateCUDAMemory() {
    cudaMalloc((void**)&imagePtr_CUDA, size_of_arr_bytes);
}

void generateMandlebrotImage() {

    // Generate image
    mandelbrot_kernel << <blocksPerGrid, threadsPerBlock >> > (imagePtr_CUDA, arr_width, arr_height);
    // Copy array to host(CPU) memory
    cudaMemcpy(imagePtr, imagePtr_CUDA, size_of_arr_bytes, cudaMemcpyDeviceToHost);

}

int main() {

    allocateCUDAMemory();
    generateMandlebrotImage();

    // Output the image to the console (for debugging)
    for (int y = 0; y < arr_height; ++y) {
        for (int x = 0; x < arr_width; ++x) {
            std::cout << imagePtr[y * arr_width + x] << " ";
        }
        std::cout << "\n";
    }

    cleanupMemory();

    return 0;
}