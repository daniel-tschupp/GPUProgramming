#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <stdio.h>
#include <math.h>
#include <iostream>

#include <cuda.h>
#include "device_launch_parameters.h"
#include "Stopwatch.h"
#include "cuda_devicehelper.h"

using namespace std;

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define satMin(x, min)   x<min?min:x;
#define ARCH 30

void removeBackground_cachingBG(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height);
void removeBackground_cachingBG_Mapped(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height);

#endif // CUDA_KERNEL_H
