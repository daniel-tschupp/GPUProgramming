#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>

#include <cuda.h>
#include "device_launch_parameters.h"
#include "Stopwatch.h"

using namespace std;

typedef struct _data{
    float* inMat1;
    float* inMat2;
    float* outMat;
    int matSize;
    int matWidth;
}Data;

void readMatrix(float* const pM, const int size_M, const string filename);
void createMatrix(float* const pM, const int size_M, const string filename);
void showMatrix(const float* const pM, const int columns, const int rows);

__global__ void matrixMulKernelStruct(Data data){
    float* M_D = data.inMat1;
    float* N_D = data.inMat2;
    float* P_D = data.outMat;
    int width = data.matWidth;

    int elementColumn = threadIdx.x;
    int elementRow = threadIdx.y;
    float elementSum = 0;

    for(int i = 0; i<width; i++){
        float Melement = M_D[elementRow * width + i];
        float Nelement = N_D[i * width + elementColumn];
        elementSum += Melement + Nelement;
    }
    P_D[elementRow * width + elementColumn] = elementSum;
}

__global__ void matrixMulKernel(float* M_D, float* N_D, float* P_D, int width){

    int elementColumn = threadIdx.x;
    int elementRow = threadIdx.y;
    float elementSum = 0;
#pragma unroll
    for(int i = 0; i<width; i++){           // unrolling improves loops because no jumps and better piplining
        float Melement = M_D[elementRow * width + i];
        float Nelement = N_D[i * width + elementColumn];
        elementSum += Melement + Nelement;
    }
    P_D[elementRow * width + elementColumn] = elementSum+1;
}

int main()
{
    // Describe matrix to create
    const int columns = 5;
    const int rows = 5;

    // Read matrix M from File and show it on the console
    const string filenameA = "matrices/Matrix_5x5_A.txt";
    float M_H[columns*rows];
    readMatrix(M_H, columns*rows, filenameA);
    showMatrix(M_H, columns, rows);

    // Read matrix M from File and show it on the console
    const string filenameB = "matrices/Matrix_5x5_B.txt";
    float N_H[columns*rows];
    readMatrix(N_H, columns*rows, filenameB);
    showMatrix(N_H, columns, rows);

    // Create Output Matrix
    float P_H1[columns*rows];
    float P_H2[columns*rows];

    // Create Device (GPU) RAM pointer
    float *M_D=nullptr, *N_D=nullptr, *P_D2=nullptr;
    Data data_D;
    data_D.matSize = columns*rows*sizeof(float);
    data_D.matWidth = columns;

    // Make Pinned memory out of the host data memory
    cudaHostRegister(M_H, data_D.matSize,0);
    cudaHostRegister(N_H, data_D.matSize,0);
    cudaHostRegister(P_H1, data_D.matSize,0);
    cudaHostRegister(P_H2, columns*rows*sizeof(float),0);
    P_H1[0] = 0;
    P_H2[0] = 0;
    // Allocate Memory on GPU
    int D_Size = columns*rows*sizeof(float);

    float *tmp1, *tmp2, *tmp3;                  // cudaMalloc doesn't work with struct so we need temporary variables.
    cudaMalloc((void**) &tmp1, data_D.matSize);
    cudaMalloc((void**) &tmp2, data_D.matSize);
    cudaMalloc((void**) &tmp3, data_D.matSize);
    data_D.inMat1 = tmp1;
    data_D.inMat2 = tmp2;
    data_D.outMat = tmp3;

    cudaMalloc((void**) &M_D, D_Size);
    cudaMalloc((void**) &N_D, D_Size);
    cudaMalloc((void**) &P_D2, D_Size);

    // Create two concurrent streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create Stopwatch to measure time
    Stopwatch sw;
    sw.Start();

    // Copy Matrices to GPU memory for both streams
    cudaMemcpyAsync(data_D.inMat1, M_H, data_D.matSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(data_D.inMat2, N_H, data_D.matSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(M_D, M_H, D_Size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(N_D, N_H, D_Size, cudaMemcpyHostToDevice, stream2);

    // Execute kernel on data
    int width = columns;
    dim3 dimBlock(width, width);  // number of theads per block
    dim3 dimGrid(1);    // number of blocks per grig
    matrixMulKernelStruct<<<dimGrid, dimBlock, 0, stream1>>>(data_D); // struct must be by value! (Because Host Pointers are not accessible from within GPU)
    matrixMulKernel<<<dimGrid, dimBlock, 0, stream2>>>(M_D, N_D, P_D2, width);

    // Read back results from GPU memory from both streams
    cudaMemcpyAsync(P_H1, data_D.outMat, data_D.matSize, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(P_H2, P_D2, D_Size, cudaMemcpyDeviceToHost, stream2);

    cudaDeviceSynchronize(); // wait until the work of all streams is done
    sw.Stop();

    // Free GPU memory
    cudaFree(data_D.inMat1);
    cudaFree(data_D.inMat2);
    cudaFree(data_D.outMat);
    cudaFree(M_D);
    cudaFree(N_D);
    cudaFree(P_D2);

    // Unregister Pinned memory
    cudaHostUnregister(M_H);
    cudaHostUnregister(N_H);
    cudaHostUnregister(P_H1);
    cudaHostUnregister(P_H2);

    // Show result matrix
    showMatrix(P_H1, columns, rows);
    showMatrix(P_H2, columns, rows);
    cout << "Time: " << sw.GetElapsedTimeMilliseconds() << endl;
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
