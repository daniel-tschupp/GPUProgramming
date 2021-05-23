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
#include "cuda_profiler_api.h"

using namespace std;

void loadData(MatrixMulData& data, CUDA_DeviceHelper& devHelper);
float matrixMulTraditional(float* M, float* N, float* P, int Width, Stopwatch& sw);
void matrixMulKernel05(MatrixMulData& data);
void matrixMulKernel06A(MatrixMulData& data);
void matrixMulKernel06B(MatrixMulData& data);
void matrixMulKernel06C(MatrixMulData& data);

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

    // Traditional way calculation for comparison
    Stopwatch swTraditional;
    float LastElementTaditional = matrixMulTraditional(data.M_H, data.N_H, data.P_H, data.Width, swTraditional);
    cout << "Traditional Way: " << swTraditional.GetElapsedTimeMilliseconds() << "ms. Value: " << LastElementTaditional << endl;

    // Executing Kernel 06
    cudaProfilerStart();
    matrixMulKernel05(data);	// Global memory acces all the time (bursts half time)
    matrixMulKernel06A(data);	// shared memory with bursts for both tiles
    matrixMulKernel06B(data);	// shared memory with bursts for one tile
    matrixMulKernel06C(data);	// shared memory with no bursts
    cudaProfilerStop();

    // Clean up
    delete data.M_H;
    delete data.N_H;
    delete data.P_H;
    //devHelper.checkDimConfigs(dimGrid, dimBlock);
    return 0;
}
void matrixMulKernel05(MatrixMulData& data){
    Stopwatch sw;


    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    size_t sharedMemSize = 2*actTileWidth*actTileWidth*sizeof(float);
    sw.Start();
    matrixMulKernel_05<<<dimGrid, dimBlock, sharedMemSize>>>(data.M_D, data.N_D, data.P_D, data.Width);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel 05: " << sw.GetElapsedTimeMilliseconds() << "ms. Value: " << data.P_H[data.Width*data.Width-1] << endl;
}
void matrixMulKernel06A(MatrixMulData& data){
    Stopwatch sw;


    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    size_t sharedMemSize = 2*actTileWidth*actTileWidth*sizeof(float);
    sw.Start();
    matrixMulKernel_06A<<<dimGrid, dimBlock, sharedMemSize>>>(data.M_D, data.N_D, data.P_D, data.Width, actTileWidth);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel 06A: " << sw.GetElapsedTimeMilliseconds() << "ms. Value: " << data.P_H[data.Width*data.Width-1] << endl;
}
void matrixMulKernel06B(MatrixMulData& data){
    Stopwatch sw;


    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    size_t sharedMemSize = 2*actTileWidth*actTileWidth*sizeof(float);
    sw.Start();
    matrixMulKernel_06B<<<dimGrid, dimBlock, sharedMemSize>>>(data.M_D, data.N_D, data.P_D, data.Width, actTileWidth);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time

    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel 06B: " << sw.GetElapsedTimeMilliseconds() << "ms. Value: " << data.P_H[data.Width*data.Width-1] << endl;
}
void matrixMulKernel06C(MatrixMulData& data){
    Stopwatch sw;

    // Initialize GPU (allocate memory & transfer data)
    cudaInitMatMulKernel(&data);

    // Execute kernel on data
    int actTileWidth = 32;
    int BlockWidth = actTileWidth;
    int GridWidth = data.Width/BlockWidth;
    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    size_t sharedMemSize = 2*actTileWidth*actTileWidth*sizeof(float);
    sw.Start();
    matrixMulKernel_06C<<<dimGrid, dimBlock, sharedMemSize>>>(data.M_D, data.N_D, data.P_D, data.Width, actTileWidth);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    sw.Stop();
    // read out gpu memory and free it.
    cudaFinalizeMatMulKernel(&data);
    cout << "Kernel 06C: " << sw.GetElapsedTimeMilliseconds() << "ms. Value: " << data.P_H[data.Width*data.Width-1] << " Slower because no burst transfers!"<< endl;
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

float matrixMulTraditional(float* M, float* N, float* P, int Width, Stopwatch& sw){
    sw.Start();
    for(int iOutCol = 0; iOutCol < Width; iOutCol++){
        for(int iOutRow = 0; iOutRow < Width; iOutRow++){
            for(int iDot = 0; iDot< Width; iDot++){
                P[iOutRow*Width + iOutCol] += M[iOutCol*Width + iDot] * N[iDot*Width + iOutCol];
            }
        }
    }
    sw.Stop();
    return P[Width*Width-1];
}
