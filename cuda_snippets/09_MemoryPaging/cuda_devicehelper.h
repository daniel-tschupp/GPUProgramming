#ifndef CUDA_DEVICEINFOHELPER_H
#define CUDA_DEVICEINFOHELPER_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
//extern void cudaGetDeviceProperties(cudaDeviceProp*, int);

class CUDA_DeviceHelper
{
    int mDeviceID;
    int mNumWarpSheduler;
    int mResidentBlocksInSM;
    cudaDeviceProp mDevProps;
public:
    CUDA_DeviceHelper(int DeviceID, bool print);
    void printDeviceData(void) const;
    const cudaDeviceProp& getDevProps(void) const;
    int getGPUFrequency(void) const;
    int getNumberOfSMs(void) const;
    int getNumberOfConcurrentKernels(void) const;
    int getWarpSize() const;
    int getNumberOfCuncurrentWarpsPerSM(void) const;
    int getGlobMemorySize(void) const;
    int getSharedMemorySize(void) const;
    int getRegisterSize(int numUsedThreadsPerSM) const;
    int getMaxThreadsPerBlock(void) const;
    int getMaxThreadsPerSM(void) const;
    string getName(void) const;
    int getNumberOfBurstRequestsPerWarpSheduler(int TileWidth) const;
    int calcParallelGlobMemRequest(int TileWidth)const;
    void checkDimConfigs(dim3 gridDim, dim3 blockDim) const;

    void readMatrix(float* const pM, const int size_M, const string filename);
    void createMatrix(float* const pM, const int size_M, const string filename);
    void showMatrix(const float* const pM, const int columns, const int rows);
};

#endif // CUDA_DEVICEINFOHELPER_H
