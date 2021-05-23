#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include "../cudaincludes.h"
//#include <cuda.h>
//#include "device_launch_parameters.h"
//#include <cuda_runtime_api.h>

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define satMin(x, min)   x<min?min:x;

float* pinHostAndMallocDeviceMemory(float* ptr, size_t size);
void pinHostMemory(float* hostPtr, size_t size);
float* mallocDeviceMemory(size_t size);
cudaError_t cpyAsyncToDevice(float* devPtr, float* hostPtr, size_t size, cudaStream_t* stream);
cudaError_t cpyAsyncToHost(float* devPtr, float* hostPtr, size_t size, cudaStream_t* stream);
void unpinHostAndFreeDeviceMemory(float* devPtr, float* hostPtr);
void unpinHostMemory(float* hostPtr);
void freeDeviceMemory(float* devPtr);
void waitForStream(cudaStream_t* stream);
cudaStream_t* getNewStream(void);
void cleanUpStream(cudaStream_t* stream);

int getNumberOfCUDADevices(void);
void getDeviceProperties(cudaDeviceProp* props, int deviceID);


#endif // KERNEL_UTILS_H
