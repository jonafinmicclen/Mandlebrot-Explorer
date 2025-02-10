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

    int neighbour_sum = 0;
    int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int sumX = 0;
    int sumY = 0;

    for (int iX = -1; iX <= 1; ++iX) {
        for (int iY = -1; iY <= 1; ++iY) {
            int sampleX = min(max(x + iX, 0), width - 1);
            int sampleY = min(max(y + iY, 0), height - 1);
            int pixel = imagePtr[sampleY * width + sampleX];

            sumX += pixel * Gx[iY + 1][iX + 1];
            sumY += pixel * Gy[iY + 1][iX + 1];
        }
    }

    // Compute gradient magnitude
    neighbour_sum = sqrtf(sumX * sumX + sumY * sumY);
    int zoom_factor = (5 * (sqrt(x_zoom) / 10) + 1000) + (neighbour_sum * neighbour_sum * sqrt(x_zoom)/20000);

    *pixelPtr = mandlebrot_iterate(C, zoom_factor);
}
