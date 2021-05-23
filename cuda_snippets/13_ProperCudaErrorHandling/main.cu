#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include "memory.h"

#define CUDA_ERROR_CHECK

#include "cudaincludes.h"
#include "imagedata.h"
#include "cuda_profiler_api.h"
#include "Kernels/kernel_utils.h"

using namespace std;

#define SIZE 4096

__global__ void vecKernel(float* data){
    uint bx = blockIdx.x;
    uint tx = threadIdx.x;
    uint col = bx*blockDim.x+tx;
    data[col] = data[col] + 1.0;
}

#define USE_ERROR_HANDLING 1

int main()
{
    // Allocating normal Memory and changing it to paging:
    cudaFree(nullptr);

    string path = "bscan512x640.png";

    float inputData[SIZE];

    for(uint i = 0; i<SIZE; i++)
        inputData[i] = 666.0;

    cout << "Start Array contains values: " << inputData[0] << endl;

    float* inData_D = pinHostAndMallocDeviceMemory(inputData, SIZE*sizeof(float));


    if(USE_ERROR_HANDLING)
        CudaSafeAPICall(cudaMemcpy(nullptr, inputData, SIZE*sizeof(float), cudaMemcpyHostToDevice));
    else
        cudaMemcpy(nullptr, inputData, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    uint maxThreadsPerBlock = 2048;
    dim3 dimGrid((float)SIZE/maxThreadsPerBlock + 0.5, 1, 1);
    dim3 dimBlock(maxThreadsPerBlock, 1, 1);

    vecKernel<<<dimGrid, dimBlock>>>(inData_D);
    if(USE_ERROR_HANDLING)
        CudaCheckKernelError();
    if(USE_ERROR_HANDLING)
        CudaSafeAPICall(cudaMemcpy(inputData, inData_D, SIZE*sizeof(float), cudaMemcpyDeviceToHost));
    else
        cudaMemcpy(inputData, inData_D, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    unpinHostAndFreeDeviceMemory(inData_D, inputData);

    cout << "New Array contains values: " << inputData[0] << endl;
}

