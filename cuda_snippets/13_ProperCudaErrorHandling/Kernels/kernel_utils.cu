#include "kernel_utils.h"

void pinHostMemory(float* hostPtr, size_t byteSize){
    CudaSafeAPICall(cudaHostRegister(hostPtr,static_cast<uint>(byteSize), cudaHostRegisterPortable|cudaHostRegisterMapped));
}
float* mallocDeviceMemory(size_t byteSize){
    float *devPtr = nullptr;
    CudaSafeAPICall(cudaMalloc((void**)&devPtr, byteSize));
    return devPtr;
}

float* pinHostAndMallocDeviceMemory(float* ptr, size_t byteSize){
    float *devPtr = nullptr;
    CudaSafeAPICall(cudaHostRegister(ptr,static_cast<uint>(byteSize), cudaHostRegisterPortable|cudaHostRegisterMapped));
    CudaSafeAPICall(cudaMalloc((void**)&devPtr, byteSize));
    return devPtr;
}
void cpyAsyncToDevice(float* devPtr, float* hostPtr, size_t byteSize, cudaStream_t* stream){
    CudaSafeAPICall(cudaMemcpyAsync(devPtr, hostPtr, byteSize, H2D, *stream));
}

void cpyAsyncToHost(float* devPtr, float* hostPtr, size_t byteSize, cudaStream_t* stream){
    CudaSafeAPICall(cudaMemcpyAsync(hostPtr, devPtr, byteSize, D2H, *stream));
}

void unpinHostMemory(float* hostPtr){
    CudaSafeAPICall(cudaHostUnregister(hostPtr));
}
void freeDeviceMemory(float* devPtr){
    CudaSafeAPICall(cudaFree(devPtr));
}
void unpinHostAndFreeDeviceMemory(float* devPtr, float* hostPtr){
    CudaSafeAPICall(cudaHostUnregister(hostPtr));
    CudaSafeAPICall(cudaFree(devPtr));
}

void waitForStream(cudaStream_t* stream){
    CudaSafeAPICall(cudaStreamSynchronize(*stream));
}

cudaStream_t* getNewStream(void){
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    CudaSafeAPICall(cudaStreamCreate(stream));
    return stream;
}

void cleanUpStream(cudaStream_t* stream){
    CudaSafeAPICall(cudaStreamDestroy(*stream));
}


int getNumberOfCUDADevices(void){
    int dev_count;
    CudaSafeAPICall(cudaGetDeviceCount(&dev_count));
    return dev_count;
}

void getDeviceProperties(cudaDeviceProp* props, int deviceID){
    CudaSafeAPICall(cudaGetDeviceProperties(props, deviceID));
}

