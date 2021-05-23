#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include "Stopwatch.h"

#include <cuda.h>
#include <cufft.h>

using namespace std;

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

void readMatrix(float* const pM, const int size_M, const string filename);
void createMatrix(float* const pM, const int size_M, const string filename);
void showMatrix(const float* const pM, const int columns, const int rows);

__global__ void matrixEAddKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){
        P_D[offset] = M_D[offset] + N_D[offset];        // 2 ld (100cycles) + 1 fmul
    }
}

__global__ void matrixESubKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){
        P_D[offset] = M_D[offset] - N_D[offset];        // 2 ld (100cycles) + 1 fmul
    }
}

__global__ void matrixEMulKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){
        P_D[offset] = M_D[offset] * N_D[offset];        // 2 ld (100cycles) + 1 fmul
    }
}

__global__ void matrixEDivKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){
        P_D[offset] = M_D[offset] * N_D[offset];       // 2 ld (100cycles) + 1 fmul
    }
}                                                      // = compute-to-global-memory-access ratio: 1:2 = 0.5 <== very bad
__global__ void matrixMulKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    // Calc Actual x, y Values of output element to calculate
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate position on array where this element is
    int offset = Row * MatrixWidth + Col;

    // Check whether the element is realy inside the matrix (in case there are to many threads
    if(Col < MatrixWidth && Row < MatrixWidth){

        // Do dot product of input matrices for output element
        float dotSum = 0;
        for(int i = 0; i<MatrixWidth;i++){
            float RowElement = M_D[Row * MatrixWidth + i];  // 1 ld (100cycles to grab element in global memory)
            float ColElement = N_D[i * MatrixWidth + Col];  // 1 ld (100cycles to grab element in global memory)
            dotSum += (RowElement * ColElement);            // 1 fadd + 1 fmul
        }

        //Insert calculated dot product inside output matrix
        P_D[offset] = dotSum;
    }
}                                                           // = compute-to-global-memory-access ratio: 2:2 = 1.0 <== still very bad

int main()
{
    // Describe matrix to create
    const int columns = 512;
    const int rows = 512;
    ifstream in;

    // Read matrix M from File and show it on the console
    const string filenameA = "matrices/Matrix_512x512_A.txt";
    float M_H[columns*rows];
    readMatrix(M_H, columns*rows, filenameA);

    //showMatrix(M_H, columns, rows);

    // Read matrix M from File and show it on the console
    const string filenameB = "matrices/Matrix_512x512_B.txt";
    float N_H[columns*rows];
    readMatrix(N_H, columns*rows, filenameB);
    //showMatrix(N_H, columns, rows);

    // Create Output Matrix
    float P_H[columns*rows];

    // Create Device (GPU) RAM pointer
    float *M_D=nullptr, *N_D=nullptr, *P_D=nullptr;

    Stopwatch sw;

    // Allocate Memory on GPU
    int D_Size = columns*rows*sizeof(float);
    cudaMalloc((void**) &M_D, D_Size);
    cudaMalloc((void**) &N_D, D_Size);
    cudaMalloc((void**) &P_D, D_Size);

    // Copy Matrices to GPU memory
    cudaMemcpy(M_D, M_H, D_Size, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(N_D, N_H, D_Size, H2D);

    // Execute kernel on data

    int warpSize = 32;
    int numWarpsPerBlock = 32;             // The more wraps the better because just one SM and very long access times to global memmory

    int MatrixWidth = sqrt(columns*rows);
    int numThreadPerBlock = warpSize * numWarpsPerBlock;
    int numBlocksPerGrid = ceil((MatrixWidth*MatrixWidth) / numThreadPerBlock);
    int GridWidth = ceil(sqrt(numBlocksPerGrid));
    int BlockWidth = ceil(sqrt(numThreadPerBlock));

    dim3 dimBlock(BlockWidth, BlockWidth, 1); // number of theads per block
    dim3 dimGrid(GridWidth, GridWidth, 1);    // number of blocks per grig

    sw.Start();
    //matrixEAddKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, MatrixWidth);
    //matrixESubKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, MatrixWidth);
    //matrixEMulKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, MatrixWidth);
    //matrixEDivKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, MatrixWidth);
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, MatrixWidth);
    sw.Stop();

    // Read back results from GPU memory
    cudaMemcpy(P_H, P_D, D_Size, D2H);

    // Free GPU memory
    cudaFree(M_D);
    cudaFree(N_D);
    cudaFree(P_D);

    // Show result matrix
    showMatrix(P_H, columns, rows);

    cout << "WarpSize: " << warpSize << endl;
    cout << "numWarpsPerBlock: " << numWarpsPerBlock << endl;
    cout << "numBlocksPerGrid: " << numBlocksPerGrid << endl;
    cout << "numThreadPerBlock: " << numThreadPerBlock << endl;
    cout << "GridWidth: " << GridWidth << endl;
    cout << "BlockWidth: " << BlockWidth << endl;
    cout << "MatrixWidth: " << MatrixWidth << endl;
    cout << "Algorithm took: " << sw.GetElapsedTimeMilliseconds() << "ms to complete." << endl;
   // cout << "Cuda Error: " << cudaGetErrorString(err) << endl;
    return 0;
}

void readMatrix(float* const pM, const int size_M, const string filename){
    ifstream inputData(filename);
    for(int i = 0; i<size_M; i++){
        string tmp;
        inputData >> tmp;
        pM[i] = std::stof(tmp);
    }
}

void createMatrix(float* const pM, const int size_M, const string filename){
    srand((unsigned)time(0));
    for(int i= 0; i < size_M; i++)
        pM[i] = static_cast <float> (rand())/static_cast <float>(20000);

    ofstream arrayData(filename); // File Creation(on C drive)

    for(int k=0;k<size_M;k++)
    {
        arrayData<<pM[k]<<endl;
    }
    cout << "Created File: " << filename << " with " << size_M << " Elements." << endl;
}

void showMatrix(const float* const pM, const int columns, const int rows){
    cout << "Matrix:" << endl;

    for(int i = 0; i < rows; i++){
        for(int k = 0; k < columns; k++)
            cout << pM[columns*i + k] << "\t";
        cout << endl;
    }
}
