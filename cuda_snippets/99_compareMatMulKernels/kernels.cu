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

__global__ void matrixMulKernel_05(float* M_D, float* N_D, float* P_D, int MatrixWidth){
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

__global__ void matrixMulKernel_06A(float* M_D, float* N_D, float* P_D, int MatrixWidth, int TileWidth){
    // Allocate shared memory recourses for the tiles
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + (TileWidth*TileWidth);

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;

    float dotSum = 0;
    // Check whether the element is realy inside the matrix (in case there are to many threads
    for(int ph=0; ph < ceil(MatrixWidth/(float)TileWidth); ph++){

        if( (Row < MatrixWidth) && (ph*TileWidth+tx < MatrixWidth)){
            Mds[ty*TileWidth + tx] = M_D[Row*MatrixWidth + ph*TileWidth + tx];      // good because of corner tuning (using burst transfers)
            //Mds[tx*TileWidth + ty] = M_D[(ph*TileWidth+ty)*MatrixWidth + Col];        // bad becaus no global memory burst transfers are possible
        }

        if( Col < MatrixWidth && (ph*TileWidth+ty) < MatrixWidth){
            Nds[ty*TileWidth + tx] = N_D[(ph*TileWidth+ty)*MatrixWidth + Col];
            //Nds[tx*TileWidth + ty] = N_D[Row*MatrixWidth + ph*TileWidth + tx];
        }
        __syncthreads();
        for( int i = 0; i < TileWidth; i++){
            dotSum += Mds[ty*TileWidth + i] * Nds[i*TileWidth + tx];
        }
        __syncthreads();

    }

    // Calculate position on array where this element is
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){

        //Insert calculated dot product inside output matrix
        P_D[offset] = dotSum;
    }
}                                                           // = compute-to-global-memory-access ratio: 2:2 = 1.0 <== still very bad

__global__ void matrixMulKernel_06B(float* M_D, float* N_D, float* P_D, int MatrixWidth, int TileWidth){
    // Allocate shared memory recourses for the tiles
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + (TileWidth*TileWidth);

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;

    float dotSum = 0;
    // Check whether the element is realy inside the matrix (in case there are to many threads
    for(int ph=0; ph < ceil(MatrixWidth/(float)TileWidth); ph++){

        int Mx = ph*TileWidth + tx;
        int My = by*TileWidth + ty;
        if(Mx < MatrixWidth && My < MatrixWidth)
            Mds[ty*TileWidth + tx] = M_D[My*MatrixWidth+Mx];

        int Nx = bx*TileWidth + tx;
        int Ny = ph*TileWidth + ty;
        if(Nx < MatrixWidth && My < MatrixWidth)
            Nds[ty*TileWidth + tx] = N_D[Ny*MatrixWidth+Nx];

        __syncthreads();
        for( int i = 0; i < TileWidth; i++){
            dotSum += Mds[ty*TileWidth + i] * Nds[i*TileWidth + tx];
        }
        __syncthreads();

    }

    // Calculate position on array where this element is
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){

        //Insert calculated dot product inside output matrix
        P_D[offset] = dotSum;
    }
}                                                           // = compute-to-global-memory-access ratio: 2:2 = 1.0 <== still very bad

__global__ void matrixMulKernel_06C(float* M_D, float* N_D, float* P_D, int MatrixWidth, int TileWidth){
    // Allocate shared memory recourses for the tiles
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + (TileWidth*TileWidth);

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;

    float dotSum = 0;
    // Check whether the element is realy inside the matrix (in case there are to many threads
    for(int ph=0; ph < ceil(MatrixWidth/(float)TileWidth); ph++){

        int Mx = ph*TileWidth + ty;
        int My = by*TileWidth + tx;
        if(Mx < MatrixWidth && My < MatrixWidth)
            Mds[tx*TileWidth + ty] = M_D[My*MatrixWidth+Mx];

        int Nx = bx*TileWidth + ty;
        int Ny = ph*TileWidth + tx;
        if(Nx < MatrixWidth && My < MatrixWidth)
            Nds[tx*TileWidth + ty] = N_D[Ny*MatrixWidth+Nx];

        __syncthreads();
        for( int i = 0; i < TileWidth; i++){
            dotSum += Mds[ty*TileWidth + i] * Nds[i*TileWidth + tx];
        }
        __syncthreads();

    }

    // Calculate position on array where this element is
    int offset = Row * MatrixWidth + Col;
    if(Col < MatrixWidth && Row < MatrixWidth){

        //Insert calculated dot product inside output matrix
        P_D[offset] = dotSum;
    }
}                                                           // = compute-to-global-memory-access ratio: 2:2 = 1.0 <== still very bad
