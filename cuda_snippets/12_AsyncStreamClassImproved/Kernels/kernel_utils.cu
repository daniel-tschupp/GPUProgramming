#include "kernel_utils.h"

void pinHostMemory(float* hostPtr, size_t size){
    cudaHostRegister(hostPtr,static_cast<uint>(size), cudaHostRegisterPortable|cudaHostRegisterMapped);
}
float* mallocDeviceMemory(size_t size){
    float *devPtr = nullptr;
    cudaMalloc((void**)&devPtr, size);
    return devPtr;
}

float* pinHostAndMallocDeviceMemory(float* ptr, size_t size){
    float *devPtr = nullptr;
    cudaHostRegister(ptr,static_cast<uint>(size), cudaHostRegisterPortable|cudaHostRegisterMapped);
    cudaMalloc((void**)&devPtr, size);
    return devPtr;
}
cudaError_t cpyAsyncToDevice(float* devPtr, float* hostPtr, size_t size, cudaStream_t* stream){
    return cudaMemcpyAsync(devPtr, hostPtr, size, H2D, *stream);
}

cudaError_t cpyAsyncToHost(float* devPtr, float* hostPtr, size_t size, cudaStream_t* stream){
    return cudaMemcpyAsync(hostPtr, devPtr, size, D2H, *stream);
}

void unpinHostMemory(float* hostPtr){
    cudaHostUnregister(hostPtr);
}
void freeDeviceMemory(float* devPtr){
    cudaFree(devPtr);
}
void unpinHostAndFreeDeviceMemory(float* devPtr, float* hostPtr){
    cudaHostUnregister(hostPtr);
    cudaFree(devPtr);
}

void waitForStream(cudaStream_t* stream){
    cudaStreamSynchronize(*stream);
}

cudaStream_t* getNewStream(void){
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cudaStreamCreate(stream);
    return stream;
}

void cleanUpStream(cudaStream_t* stream){
    cudaStreamDestroy(*stream);
}


int getNumberOfCUDADevices(void){
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    return dev_count;
}

void getDeviceProperties(cudaDeviceProp* props, int deviceID){
    cudaGetDeviceProperties(props, deviceID);
}
