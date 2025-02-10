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
    // Calculate area of array to work on
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Distribute pixels evenly in square around 0,0
    Complex C(  x_offset + (x - (width/2))/x_zoom,
                y_offset + (y - (height/2))/y_zoom  );

    int* pixelPtr = &imagePtr[y * width + x];
   
    // Iterations optimiser, use convolution of near pixel to calculate max iterations frame will converge to true mandlebrot 
    

    int neighbour_sum = 0;
    int i = 0;
    int convolution_width = 2;
    for (int iX = max(x - convolution_width, 0); iX <= min(x + convolution_width, width); ++iX) {
        for (int iY = max(y - convolution_width, 0); iY <= min(y + convolution_width, height); ++iY)
        {
            neighbour_sum += imagePtr[iY * width + iX] * imagePtr[iY * width + iX];
            ++i;
        }
    }
    neighbour_sum = sqrtf(neighbour_sum);
    int zoom_factor = 5 * (neighbour_sum * sqrt(x_zoom) / 10) + 1000;

    *pixelPtr = mandlebrot_iterate(C, zoom_factor);
}
