#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>

#include <cuda.h>
#include <cufft.h>

using namespace std;

void readMatrix(float* const pM, const int size_M, const string filename);
void createMatrix(float* const pM, const int size_M, const string filename);
void showMatrix(const float* const pM, const int columns, const int rows);

__global__ void matrixMulKernel(float* M_D, float* N_D, float* P_D, int width){
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
    float P_H[columns*rows];

    // Create Device (GPU) RAM pointer
    float *M_D=nullptr, *N_D=nullptr, *P_D=nullptr;

    // Allocate Memory on GPU
    int D_Size = columns*rows*sizeof(float);
    cudaMalloc((void**) &M_D, D_Size);
    cudaMalloc((void**) &N_D, D_Size);
    cudaMalloc((void**) &P_D, D_Size);

    // Copy Matrices to GPU memory
    cudaMemcpy(M_D, M_H, D_Size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_D, N_H, D_Size, cudaMemcpyHostToDevice);

    // Execute kernel on data
    int width = columns;
    dim3 dimBlock(width, width);  // number of theads per block
    dim3 dimGrid(1);    // number of blocks per grig
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_D, N_D, P_D, width);

    // Read back results from GPU memory
    cudaMemcpy(P_H, P_D, D_Size, cudaMemcpyDeviceToHost);

    // Wait for operation to complete
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(M_D);
    cudaFree(N_D);
    cudaFree(P_D);

    // Show result matrix
    showMatrix(P_H, columns, rows);

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
