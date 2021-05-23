/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_utils.h"
#include "cuda_errorchecking.h"
#include <stdio.h>
#include <assert.h>


#ifdef DEBUG_GLOB_ALLOC_MEM
    int allocGlobMem = 0;
#endif
/**********************************************************************************************************************************************************/
/****************************************************************       allocation        *****************************************************************/
/**********************************************************************************************************************************************************/

void gpuPinHostMemory(void* hostPtr, size_t byteSize){
    CudaSafeAPICall(cudaHostRegister(hostPtr,static_cast<unsigned int>(byteSize), cudaHostRegisterPortable|cudaHostRegisterMapped));
}
void* gpuMallocDeviceMemory(size_t byteSize){
    void* devPtr = nullptr;
#ifdef DEBUG_GLOB_ALLOC_MEM
    allocGlobMem += byteSize;
    printf("Total allocated global memory: %g Mbytes\n", ((float)allocGlobMem/1000000));
#endif
    CudaSafeAPICall(cudaMalloc((void**)&devPtr, byteSize));
    CudaSafeAPICall(cudaMemset(devPtr, 0, byteSize));
    assert(devPtr != nullptr);
    return devPtr;
}

void* gpuPinHostAndMallocDeviceMemory(void* ptr, size_t byteSize){
    void* devPtr = nullptr;
#ifdef DEBUG_GLOB_ALLOC_MEM
    allocGlobMem += byteSize;
    printf("Total allocated global memory: %g Mbytes\n", ((float)allocGlobMem/1000000));
#endif
    CudaSafeAPICall(cudaHostRegister(ptr,static_cast<unsigned int>(byteSize), cudaHostRegisterPortable|cudaHostRegisterMapped));
    CudaSafeAPICall(cudaMalloc((void**)&devPtr, byteSize));
    CudaSafeAPICall(cudaMemset(&devPtr, 0, byteSize));
    assert(devPtr != nullptr);
    return devPtr;
}

void gpuUnpinHostMemory(void* hostPtr){
    CudaSafeAPICall(cudaHostUnregister(hostPtr));
}
void gpuFreeDeviceMemory(void* devPtr){
    CudaSafeAPICall(cudaFree(devPtr));
#ifdef DEBUG_GLOB_ALLOC_MEM
    allocGlobMem = 0;
#endif
}
void gpuUnpinHostAndFreeDeviceMemory(void* devPtr, void* hostPtr){
    CudaSafeAPICall(cudaHostUnregister(hostPtr));
    CudaSafeAPICall(cudaFree(devPtr));
}

/*********************************************************************************************************************************************************/
/*******************************************************************       copy        *******************************************************************/
/*********************************************************************************************************************************************************/

void gpuCpyAsyncToDevice(void* devPtr, const void* hostPtr, size_t byteSize, cudaStream_t* stream, bool sync){
    CudaSafeAPICall(cudaMemcpyAsync(devPtr, hostPtr, byteSize, H2D, *stream));
    if(sync)
        CudaSafeAPICall(cudaStreamSynchronize(*stream));
}

void gpuCpyAsyncToHost(const void* devPtr, void* hostPtr, size_t byteSize, cudaStream_t* stream, bool sync){
    CudaSafeAPICall(cudaMemcpyAsync(hostPtr, devPtr, byteSize, D2H, *stream));
    if(sync)
        CudaSafeAPICall(cudaStreamSynchronize(*stream));
}

void gpuCpySyncToDevice(void* devPtr, const void* hostPtr, size_t byteSize){
    CudaSafeAPICall(cudaMemcpy(devPtr, hostPtr, byteSize, H2D));
}

void gpuCpySyncToHost(const void* devPtr, void* hostPtr, size_t byteSize){
    CudaSafeAPICall(cudaMemcpy(hostPtr, devPtr, byteSize, D2H));
}
/**********************************************************************************************************************************************************/
/*******************************************************************       streams        *****************************************************************/
/**********************************************************************************************************************************************************/

void gpuWaitForStream(cudaStream_t* stream){
    CudaSafeAPICall(cudaStreamSynchronize(*stream));
}

cudaStream_t* gpuGetNewStream(void){
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    CudaSafeAPICall(cudaStreamCreate(stream));
#ifdef DEBUG_GLOB_ALLOC_MEM
    printf("Stream reference: %d\n", stream);
#endif
    return stream;
}

void gpuCleanUpStream(cudaStream_t* stream){
    CudaSafeAPICall(cudaStreamDestroy(*stream));
}

/**********************************************************************************************************************************************************/
/*******************************************************************       Device Info        *************************************************************/
/**********************************************************************************************************************************************************/

int gpuGetNumberOfCUDADevices(void){
    int dev_count;
    CudaSafeAPICall(cudaGetDeviceCount(&dev_count));
    return dev_count;
}

void gpuGetDeviceProperties(cudaDeviceProp* props, int deviceID){
    CudaSafeAPICall(cudaGetDeviceProperties(props, deviceID));
}

void gpuPrintDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n", devProp.totalConstMem);
    printf("Texture alignment:             %lu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

void gpuGetGPUInfos()
{
    int devCount;
    CudaSafeAPICall(cudaGetDeviceCount(&devCount));
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        CudaSafeAPICall(cudaGetDeviceProperties(&devProp, i));
        gpuPrintDevProp(devProp);
    }

    printf("size of cufftcomplex: %li ",sizeof(cufftComplex) );
}

#ifdef __cplusplus
}
#endif

/* Copyright details:
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
**
** Contact: Daniel Tschupp ( daniel.tschupp@gmail.com )
*/

