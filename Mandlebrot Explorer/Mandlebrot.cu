#include "Mandlebrot.cuh"

__device__ void mandlebrot_iterate(Complex C, int* imagePtr, float x_zoom, float y_zoom, float x_offset, float y_offset)
{
    Complex Zn = Complex(0, 0);
    Complex complexPoints[MAX_ITERATIONS];

    int iterations_countdown = MAX_ITERATIONS;

    while (--iterations_countdown)
    {
        if (Zn.magnitude() >= 2.0) 
        { 

            for (int i = MAX_ITERATIONS - 1; i > iterations_countdown + 1; --i) 
            {
                // Convert Complex positions back to array coordinates
                int x = complexPoints[i].real * x_zoom - x_offset + ARR_WIDTH / 2;
                int y = complexPoints[i].imag * y_zoom - y_offset + ARR_HEIGHT / 2;

                int imag_idx = y * ARR_WIDTH + x;

                // Ensure in bounds then add
                if (0 <= imag_idx < ARR_SIZE) {
                    atomicAdd(&imagePtr[imag_idx], 1);
                }
                
            }
            return;
        }

        // Store point
        complexPoints[iterations_countdown] = Zn;
        // Iterate mandlebrot function
        Zn = Zn * Zn + C;
    }
    return;

}


__global__ void mandelbrot_kernel(int* imagePtr, int width, int height,
    float x_zoom, float y_zoom, float x_offset, float y_offset)

{
    // Determine thread coordinate from block idx
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    // Calculate related Complex position from array coordinate
    Complex C(  x_offset + (x - (width/2))/x_zoom,      // Real
                y_offset + (y - (height/2))/y_zoom  );  // Imag

    mandlebrot_iterate(C, imagePtr, x_zoom, y_zoom, x_offset, y_offset);
}