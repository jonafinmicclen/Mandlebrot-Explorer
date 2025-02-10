// Local
#include "Mandlebrot.cuh"
#include "Render.h"

// Standard
#include <algorithm>
#include <iostream>

// CUDA
#include "device_launch_parameters.h"



// Image array parameters
const int arr_width = 1000;
const int arr_height = 1000;
const int size_of_arr = arr_height * arr_width;
int size_of_arr_bytes = sizeof(int) * size_of_arr;

float XZOOM = arr_width/4;
float YZOOM = arr_width/4;
float XOFFSET = 0.0f;
float YOFFSET = 0.0f;

// Allocate device and host memory
int* imagePtr = new int[size_of_arr];
int* imagePtr_CUDA;

dim3 threadsPerBlock(32, 32);
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

void generateMandlebrotImage(float x_zoom, float y_zoom, float x_offset, float y_offset) {

    // Generate image
    mandelbrot_kernel << <blocksPerGrid, threadsPerBlock >> > (imagePtr_CUDA, arr_width, arr_height, x_zoom, y_zoom, x_offset, y_offset);
    // Copy array to host(CPU) memory
    cudaMemcpy(imagePtr, imagePtr_CUDA, size_of_arr_bytes, cudaMemcpyDeviceToHost);

}

void Display() {
    generateMandlebrotImage(XZOOM, YZOOM, XOFFSET, YOFFSET);
    cudaDeviceSynchronize();
    OpenGLAbstractions::InitialiseRender();
    OpenGLAbstractions::RenderArray(imagePtr, arr_width, arr_height, 0.01f);
    OpenGLAbstractions::FinaliseRender();

}

void MouseWheel(int button, int dir, int x, int y)
{
    if (button == 3) {  // Zoom in
        XZOOM += 0.1f * XZOOM;
        YZOOM += 0.1f * YZOOM;
    }
    else if (button == 4) {  // Zoom out
        XZOOM -= 0.1f * XZOOM;
        YZOOM -= 0.1f * YZOOM;
    }
    XOFFSET -= (y - arr_width / 2) / (XZOOM * 10);
    YOFFSET += (x - arr_height / 2) / (YZOOM * 10);

    glutPostRedisplay();
}

void Timer(int value) {
    glutPostRedisplay();  // Request redraw
    glutTimerFunc(1000 / 60, Timer, 0);  // Call again in ~16.67ms
}

int main(int argc, char** argv) {

    OpenGLAbstractions::InitialiseOpenGL(arr_width, arr_height, argc, argv);

    glutDisplayFunc(Display);
    glutMouseFunc(MouseWheel);

    allocateCUDAMemory();

    glutMainLoop();

    glutTimerFunc(1000 / 60, Timer, 0);  // Call again in ~16.67ms

    cleanupMemory();

    return 0;
}

