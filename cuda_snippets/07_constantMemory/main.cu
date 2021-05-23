#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <ctime>
#include <thread>
#include <chrono>
#include "Stopwatch.h"
#include "cuda_devicehelper.h"

#include <cuda.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.cuh"

using namespace std;

void loadData(MatrixMulData& data, CUDA_DeviceHelper& devHelper);

void emptyKernel(MatrixMulData& data);
void matrixScalarMul(MatrixMulData& data);
void matrixScalarMul_cached(MatrixMulData& data);
void matrixScalarMul_constMem(MatrixMulData& data);

int main()
{
    // Analyze Graphics Card
    int DeviceID = 0;
    CUDA_DeviceHelper devHelper(DeviceID, true);
    cudaDeviceProp dev_prop = devHelper.getDevProps();
    cudaFree(0);    // Doing slow cuda start up stuff on first cuda call.

    // Describe matrix to create
    MatrixMulData data;
    data.Width = 512;

    // Load dummy values into the input matrices
    loadData(data, devHelper);

    // Executing Kernel 06
    emptyKernel(data);
    matrixScalarMul(data);
    matrixScalarMul_cached(data);
    matrixScalarMul_constMem(data);

    // Clean up
    delete data.M_H;
    delete data.N_H;
    delete data.P_H;
    //devHelper.checkDimConfigs(dimGrid, dimBlock);
    return 0;
}
void emptyKernel(MatrixMulData& data){
    Stopwatch sw;

    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    sw.Start();
    emptyKernel<<<dimGrid, dimBlock>>>(data.M_D, data.N_D, data.P_D, data.Width);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Launch empty kernel: " << sw.GetElapsedTimeMilliseconds() << "ms." << endl;
}
void matrixScalarMul(MatrixMulData& data){
    Stopwatch sw;
    for(int i=0; i<data.Width;i++){
        for(int k=0; k<data.Width;k++)
            data.N_H[i*data.Width+k]=3.1415;
    }

    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    sw.Start();
    matrixSkalarMulKernel<<<dimGrid, dimBlock>>>(data.M_D, data.N_D, data.P_D, data.Width);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel scalar mul with cached scalar: " << sw.GetElapsedTimeMilliseconds() << "ms." << endl;
}
void matrixScalarMul_cached(MatrixMulData& data){
    Stopwatch sw;
    for(int i=0; i<data.Width;i++){
        for(int k=0; k<data.Width;k++)
            data.N_H[i*data.Width+k]=3.1415;
    }

    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    float scalarH = 3.1415f;

    sw.Start();
    matrixSkalarMulKernel_Cached<<<dimGrid, dimBlock>>>(data.M_D, scalarH, data.P_D, data.Width);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel scalar mul with scalar in const memory: " << sw.GetElapsedTimeMilliseconds() << "ms." << endl;
}


extern __constant__ float scalarConst;
void matrixScalarMul_constMem(MatrixMulData& data){
    Stopwatch sw;
    for(int i=0; i<data.Width;i++){
        for(int k=0; k<data.Width;k++)
            data.N_H[i*data.Width+k]=3.1415;
    }

    // Initialize GPU (allocate memory & transfer data
    float scalar = 3.1415f;
    cudaMemcpyToSymbol(&scalarConst, &scalar, sizeof(float));
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    sw.Start();
    matrixSkalarMulKernel_constMem<<<dimGrid, dimBlock>>>(data.M_D, data.P_D, data.Width);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel scalar mul in matrix: " << sw.GetElapsedTimeMilliseconds() << "ms." << endl;
}

void loadData(MatrixMulData& data, CUDA_DeviceHelper& devHelper){
    // Read matrix M from File and show it on the console
    int elements = data.Width*data.Width;
    const string filenameA = "matrices/Matrix_512x512_A.txt";
    data.M_H = new float[elements];
    devHelper.readMatrix(data.M_H, elements, filenameA);

    // Read matrix M from File and show it on the console
    const string filenameB = "matrices/Matrix_512x512_B.txt";
    data.N_H = new float[elements];
    devHelper.readMatrix(data.N_H, elements, filenameB);

    // Create Output Matrix
    data.P_H = new float[elements];

    // Create Device (GPU) RAM pointer
    data.M_D = nullptr; data.N_D = nullptr; data.P_D = nullptr;
}

