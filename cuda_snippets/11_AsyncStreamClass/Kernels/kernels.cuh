#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "../cudaincludes.h"
#include "kernel_utils.h"

typedef struct _BackgroundKernelParams{
    float* dBScanIn;
    float* dBScanOut;
    float* dBackground;
    uint bScanWidth;
    uint bScanHeight;
    uint blockWidth;
}BackgroundKernelParams;

void executeBGRemoveKernel(cudaStream_t* stream, BackgroundKernelParams params);

#endif // CUDA_KERNEL_H
