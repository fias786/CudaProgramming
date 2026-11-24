#include "vec_add.cuh"

__global__ void vector_add_kernel(float* A, float * B, float* C, int vectorLength){
    int workIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if( workIndex < vectorLength){
        C[workIndex] = A[workIndex] + B[workIndex];
    }
};

void vector_add(float* A, float * B, float* C, int vectorLength){
    int blockSize = 256;
    int gridSize = cuda::ceil_div(vectorLength, blockSize);
    vector_add_kernel<<<gridSize, blockSize>>>(A, B, C, vectorLength);
};