#include <cmath>
#include <cuda_runtime.h>

struct Complex
{
public:
    float real;
    float imag;

    __device__ Complex(float r, float i) : real(r), imag(i) {}

    __device__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
            real * other.imag + imag * other.real);
    }

    __device__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    __device__ float magnitude() const {
        return sqrt(real * real + imag * imag);
    }
};