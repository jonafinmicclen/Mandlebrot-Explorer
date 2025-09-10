// Local
#include "Mandlebrot.cuh"
#include "Render.h"

// Standard
#include <algorithm>
#include <iostream>

// CUDA
#include "device_launch_parameters.h"

// Global constants
#include "constants.h"

// Render parameters
float BRIGHTNESS = 0.1f;                                                                     

// Image array parameters
float XZOOM = ARR_WIDTH/4;
float YZOOM = ARR_WIDTH/4;
float XOFFSET = 0.0f;
float YOFFSET = 0.0f;

// Allocate device and host memory
int* imagePtr = new int[ARR_SIZE];
int* imagePtr_CUDA;

dim3 threadsPerBlock(10, 10);
dim3 blocksPerGrid((ARR_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (ARR_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

void cleanupMemory() {
    // Cleanup memory
    cudaFree(imagePtr_CUDA);
    delete[] imagePtr;
}

void allocateCUDAMemory() {
    cudaMalloc((void**)&imagePtr_CUDA, ARR_SIZE_B);
}

void generateMandlebrotImage(float x_zoom, float y_zoom, float x_offset, float y_offset) {
    // Reset image array
    cudaMemset(imagePtr_CUDA, 0, ARR_SIZE_B);
    // Generate image
    mandelbrot_kernel << <blocksPerGrid, threadsPerBlock >> > (imagePtr_CUDA, ARR_WIDTH, ARR_HEIGHT, x_zoom, y_zoom, x_offset, y_offset);
    // Copy array to host(CPU) memory
    cudaMemcpy(imagePtr, imagePtr_CUDA, ARR_SIZE_B, cudaMemcpyDeviceToHost);

}

// Main loop
void Display() {
    generateMandlebrotImage(XZOOM, YZOOM, XOFFSET, YOFFSET);
    cudaDeviceSynchronize();
    OpenGLAbstractions::InitialiseRender();
    OpenGLAbstractions::RenderArray(imagePtr, ARR_WIDTH, ARR_HEIGHT, BRIGHTNESS);
    OpenGLAbstractions::FinaliseRender();

}

void MouseWheel(int button, int dir, int x, int y)  // x y is mouse position
{
    // Zoom image
    if (button == 3) {  // Zoom in
        XZOOM += 0.1f * XZOOM;
        YZOOM += 0.1f * YZOOM;
    }
    else if (button == 4) {  // Zoom out
        XZOOM -= 0.1f * XZOOM;
        YZOOM -= 0.1f * YZOOM;
    }

    // Pan image
    // divide by zoom to avoid increasing pan sensitivity from zooming

    XOFFSET -= (y - ARR_WIDTH / 2) / (XZOOM * 10);  
    YOFFSET += (x - ARR_HEIGHT / 2) / (YZOOM * 10);

    glutPostRedisplay();
}

int main(int argc, char** argv) {

    OpenGLAbstractions::InitialiseOpenGL(ARR_WIDTH, ARR_HEIGHT, argc, argv);

    glutDisplayFunc(Display);
    glutMouseFunc(MouseWheel);

    allocateCUDAMemory();

    glutMainLoop();

    cleanupMemory();

    return 0;
}

