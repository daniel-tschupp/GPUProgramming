#include "kernels.cuh"

void cudaInitMatMulKernel(MatrixMulData* data){
    // Allocate Memory on GPU
    int D_Size = data->Width*data->Width*sizeof(float);
    cudaMalloc((void**) &data->M_D, D_Size);
    cudaMalloc((void**) &data->N_D, D_Size);
    cudaMalloc((void**) &data->P_D, D_Size);

    // Copy Matrices to GPU memory
    cudaMemcpy(data->M_D, data->M_H, D_Size, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(data->N_D, data->N_H, D_Size, H2D);

}
void cudaFinalizeMatMulKernel(MatrixMulData* data){
    // Read back results from GPU memory
    int D_Size = data->Width*data->Width*sizeof(float);
    cudaMemcpy(data->P_H, data->P_D, D_Size, D2H);

    // Free GPU memory
    cudaFree(data->M_D);
    cudaFree(data->N_D);
    cudaFree(data->P_D);
}

__global__ void emptyKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
}

__global__ void matrixSkalarMulKernel(float* M_D, float* N_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    P_D[Row * MatrixWidth + Col] = M_D[Row * MatrixWidth + Col] * N_D[Row * MatrixWidth + Col]; // Scalar is in a matrix so nothing can be cached.
}

__global__ void matrixSkalarMulKernel_Cached(float* M_D, float scalar, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    P_D[Row * MatrixWidth + Col] = M_D[Row * MatrixWidth + Col] * scalar; // Scalar is in a matrix so nothing can be cached.
}

__constant__ float scalarConst;
__global__ void matrixSkalarMulKernel_constMem(float* M_D, float* P_D, int MatrixWidth){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    P_D[Row * MatrixWidth + Col] = M_D[Row * MatrixWidth + Col] * scalarConst; // Scalar is in a matrix so nothing can be cached.
}
