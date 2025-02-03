#include "Mandlebrot.cuh"


__device__ float mandlebrot_iterate(Complex C, int max_iterations)
{
    Complex Zn = Complex(0, 0);
    int iterations_countdown = max_iterations;

    while (--iterations_countdown)
    {
        if (Zn.magnitude() >= 2.0) { return max_iterations - iterations_countdown; }
        Zn = Zn * Zn + C;
    }
    return 0;
}

__global__ void mandelbrot_kernel(int* imagePtr, int width, int height,
    float x_zoom, float y_zoom, float x_offset, float y_offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Complex C(  x_offset + (x - (width/2))/x_zoom,
                y_offset + (y - (height/2))/y_zoom  );

    imagePtr[y * width + x] = mandlebrot_iterate(C, 1000);
}
