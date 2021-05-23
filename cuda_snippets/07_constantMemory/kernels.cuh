#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

typedef struct _MatrixMulData{
    float *M_H, *N_H, *P_H, *M_D, *N_D, *P_D;
    size_t Width;
} MatrixMulData;

/** Initialize cuda for matrix multiplication kernels */
void cudaInitMatMulKernel(MatrixMulData* data);

__global__ void emptyKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth);
__global__ void matrixSkalarMulKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth);
__global__ void matrixSkalarMulKernel_Cached(float* M_D, float scalar, float* P_D, int MatrixWidth);
__global__ void matrixSkalarMulKernel_constMem(float* M_D, float* P_D, int MatrixWidth);

/** Cleanup, frees resources used by the device. */
void cudaFinalizeMatMulKernel(MatrixMulData* data);


#endif // CUDA_KERNEL_H
