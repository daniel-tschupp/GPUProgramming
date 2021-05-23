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

using namespace std;

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

float matrixMulTraditional(float* M, float* N, float* P, int Width, Stopwatch& sw);


struct hwSignal{
    int tx, ty, bx, by, Col, Row;
};

struct complex {
   union { // anonymous union
      cufftComplex cufft_Complex;       // cufftComplex value
      struct { int x, y; };             // anonymous structure to access x, y of cufftComplex
      struct { long real, imag; };      // anonymous structure to access real and imaginary part of cufftComplex
   };
};

__global__ void matrixMulKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth, int TileWidth){
    // Allocate shared memory resources for the tiles
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + (TileWidth*TileWidth);

    // Calc Actual x, y Values of output element to calculate
    struct hwSignal sig  { threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, 0, 0 };
    sig.Col = sig.bx *TileWidth + sig.tx;
    sig.Row = sig.by * TileWidth + sig.ty;
    /*int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;*/

    float dotSum = 0;
    // Check whether the element is really inside the matrix (in case there are to many threads
    for(int ph=0; ph < ceil(MatrixWidth/(float)TileWidth); ph++){
        if( (sig.Row < MatrixWidth) && (ph*TileWidth+sig.tx < MatrixWidth)){
            Mds[sig.ty*TileWidth + sig.tx] = M_D[sig.Row*MatrixWidth + ph*TileWidth + sig.tx];      // good because of corner tuning (using burst transfers)
            //Mds[tx*TileWidth + ty] = M_D[(ph*TileWidth+ty)*MatrixWidth + Col];        // bad becaus no global memory burst transfers are possible
        }
        if( sig.Col < MatrixWidth && (ph*TileWidth+sig.ty) < MatrixWidth){
            Nds[sig.ty*TileWidth + sig.tx] = N_D[(ph*TileWidth+sig.ty)*MatrixWidth + sig.Col];
            //Nds[tx*TileWidth + ty] = N_D[Row*MatrixWidth + ph*TileWidth + tx];
        }
        __syncthreads();
        for( int i = 0; i < TileWidth; i++){
            dotSum += Mds[sig.ty*TileWidth + i] * Nds[i*TileWidth + sig.tx];
        }
        __syncthreads();

    }

    // Calculate position on array where this element is
    int offset = sig.Row * MatrixWidth + sig.Col;
    if(sig.Col < MatrixWidth && sig.Row < MatrixWidth){

        //Insert calculated dot product inside output matrix
        P_D[offset] = dotSum;
    }
}                                                           // = compute-to-global-memory-access ratio: 2:2 = 1.0 <== still very bad


int main()
{
    // Analyze Graphics Card
    int DeviceID = 0;
    CUDA_DeviceHelper devHelper(DeviceID, true);
    cudaDeviceProp dev_prop = devHelper.getDevProps();
    cudaFree(0);    // Doing slow cuda start up stuff on first cuda call.

    // Describe matrix to create
    const int columns = 512;
    const int rows = 512;
    int MatrixWidth = columns;
    // Read matrix M from File and show it on the console
    const string filenameA = "matrices/Matrix_512x512_A.txt";
    float M_H[columns*rows];
    devHelper.readMatrix(M_H, columns*rows, filenameA);
    //showMatrix(M_H, columns, rows);

    // Read matrix M from File and show it on the console
    const string filenameB = "matrices/Matrix_512x512_B.txt";
    float N_H[columns*rows];
    devHelper.readMatrix(N_H, columns*rows, filenameB);
    //showMatrix(N_H, columns, rows);

    // Create Output Matrix
    float P_H[columns*rows];

    // Create Device (GPU) RAM pointer
    float *M_D=nullptr, *N_D=nullptr, *P_D=nullptr;

    // Traditional way calculation for comparison
    Stopwatch swTraditional, swAlloc, swH2D, swKernel, swD2H, swFree;
    float LastElement = matrixMulTraditional(M_H, N_H, P_H, columns, swTraditional);

    // Allocate Memory on GPU
    int D_Size = columns*rows*sizeof(float);
    swAlloc.Start();
    cudaMalloc((void**) &M_D, D_Size);
    cudaMalloc((void**) &N_D, D_Size);
    cudaMalloc((void**) &P_D, D_Size);
    cudaDeviceSynchronize();
    swAlloc.Stop();

    // Copy Matrices to GPU memory
    swH2D.Start();
    cudaMemcpy(M_D, M_H, D_Size, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(N_D, N_H, D_Size, H2D);
    swH2D.Stop();

    // Execute kernel on data
    int actTileWidth = 16;
    int actTileSize = actTileWidth*actTileWidth;
    int BlockWidth = actTileWidth;
    int GridWidth = MatrixWidth/BlockWidth;

    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of threads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig
    size_t sharedMemSize = 2*actTileSize*sizeof(float);

    swKernel.Start();
    matrixMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(M_D, N_D, P_D, MatrixWidth, actTileWidth);
    cudaDeviceSynchronize();    // wait for kernel to finish to measure the correct time
    swKernel.Stop();

    // Read back results from GPU memory
    swD2H.Start();
    cudaMemcpy(P_H, P_D, D_Size, D2H);
    swD2H.Stop();

    // Free GPU memory
    swFree.Start();
    cudaFree(M_D);
    cudaFree(N_D);
    cudaFree(P_D);
    cudaDeviceSynchronize();
    swFree.Stop();

    // Show results
    devHelper.showMatrix(P_H, columns, rows);
    cout << "Last Element of Traditional way: " << LastElement << endl << endl;

    devHelper.checkDimConfigs(dimGrid, dimBlock);

    cout << "One Core CPU calculation: " << swTraditional.GetElapsedTimeMilliseconds() << endl;
    cout << "swAlloc:\t" << swAlloc.GetElapsedTimeMilliseconds() << "ms" << endl;
    cout << "swH2D:\t" << swH2D.GetElapsedTimeMilliseconds() << "ms" << endl;
    cout << "swKernel:\t" << swKernel.GetElapsedTimeMilliseconds() << "ms" << endl;
    cout << "swD2H:\t" << swD2H.GetElapsedTimeMilliseconds() << "ms" << endl;
    cout << "swFree:\t" << swFree.GetElapsedTimeMilliseconds() << "ms" << endl;

    int wholeCUDATime = swAlloc.GetElapsedTimeMilliseconds()+swH2D.GetElapsedTimeMilliseconds()+
            swKernel.GetElapsedTimeMilliseconds()+swD2H.GetElapsedTimeMilliseconds()+swFree.GetElapsedTimeMilliseconds();
    float parallizationFactor = swTraditional.GetElapsedTimeMilliseconds() / (float)wholeCUDATime;
    cout << "Speed Boost through GPU: " << parallizationFactor << endl;
    return 0;
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
