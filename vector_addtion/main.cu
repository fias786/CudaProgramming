#include <iostream>
#include "vec_add.cuh"

int main(int argc, char** argv) {
    int vectorLength = 1024;

    if(argc >= 2) {
        vectorLength = std::atoi(argv[1]);
    }

    size_t size = vectorLength * sizeof(float);
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    cudaMallocHost((void **)&A, size);
    cudaMallocHost((void **)&B, size);
    cudaMallocHost((void **)&C, size);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    std::fill_n(A, vectorLength, 1.0f);
    std::fill_n(B, vectorLength, 2.0f);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vector_add(d_A, d_B, d_C, vectorLength);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < vectorLength; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}