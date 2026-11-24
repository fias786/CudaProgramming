#pragma once

#include <cuda_runtime.h>
#include <cuda/cmath>

void vector_add(float* A, float * B, float* C, int vectorLength);