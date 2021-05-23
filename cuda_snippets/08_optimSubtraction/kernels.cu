#include "kernels.cuh"

// Original Kernel
__global__ void removeBG_original(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth){
    uint col = blockIdx.x*blockDim.x+threadIdx.x;
    uint row = blockIdx.y;
    uint idx = row*BScanWidth + col;
    if( (row<BScanHeigth) && (col<BScanWidth))
        outBScan[idx] = satMin(inBScan[idx] - bg[col],0);
}
void removeBackground_original(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    CUDA_DeviceHelper devHelper(0,false);
    Stopwatch sw;


    // Calc dimensions
    uint bWidth = 32*4;
    uint gWidth = ceil(width/bWidth);
    uint gHeight = height;

    dim3 dimBlock(bWidth, 1, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *bG_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    int sizeBG = width*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &bG_D, sizeBG);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

    // Execute Kernel
    removeBG_original<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, bG_D, width, height);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(bG_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Original Kernel took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}

// Improved Kernel (all in shared Memory)
__global__ void removeBG_sharedMemory(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int TileWidth){
    // Allocate shared memory recourses for the tiles
    extern __shared__ float sharedMem[];
    float* bg_Ds = sharedMem;
    float* inTile_Ds = sharedMem + TileWidth;

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;
    uint idx = Row*BScanWidth + Col;

    // Load background into shared memory
    if( Col<BScanWidth )
        bg_Ds[tx] = bg[Col];

    // Load B-Scan-Tile into shared memory
    if( (Col<BScanWidth) && (Row<BScanHeigth) )
        inTile_Ds[ty*TileWidth+tx] = inBScan[idx];

    // Calculate Ouptut Pixels
    if( (Row<BScanHeigth) && (Col<BScanWidth) )
        outBScan[idx] = satMin(inTile_Ds[ty*TileWidth+tx] - bg_Ds[tx],0);
}
void removeBackground_sharedMemory(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    CUDA_DeviceHelper devHelper(0,false);
    Stopwatch sw;

    // Calc dimensions
    uint bWidth = 32;
    uint TileSize = (1+bWidth)*bWidth*sizeof(float);
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bWidth);

    dim3 dimBlock(bWidth, bWidth, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    //devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *bG_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    int sizeBG = width*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &bG_D, sizeBG);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

    // Execute Kernel
    removeBG_sharedMemory<<<dimGrid, dimBlock, TileSize>>>(inBScan_D, outBScan_D, bG_D, width, height, bWidth);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(bG_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Shared Memory Kernel took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}

// Improved Kernel (only background in shared memory)
__global__ void removeBG_BGsharedMemory(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int TileWidth){
    // Allocate shared memory recourses for the tiles
    extern __shared__ float bg_Ds[];

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx *TileWidth + tx;
    int Row = by * TileWidth + ty;
    uint idx = Row*BScanWidth + Col;

    // Load background into shared memory
    if( Col<BScanWidth )
        bg_Ds[tx] = bg[Col];

    // Calculate Ouptut Pixels
    if( (Row<BScanHeigth) && (Col<BScanWidth) )
        outBScan[idx] = satMin(inBScan[idx] - bg_Ds[tx],0);
}
void removeBackground_BGsharedMemory(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    CUDA_DeviceHelper devHelper(0,false);
    Stopwatch sw;

    // Calc dimensions
    uint bWidth = 32;
    uint TileSize = bWidth*sizeof(float);
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bWidth);

    dim3 dimBlock(bWidth, bWidth, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    //devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *bG_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    int sizeBG = width*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &bG_D, sizeBG);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

    // Execute Kernel
    removeBG_BGsharedMemory<<<dimGrid, dimBlock, TileSize>>>(inBScan_D, outBScan_D, bG_D, width, height, bWidth);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(bG_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Background only Shared Memory Kernel took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}

// Improved Kernel (constant memory)
const uint BG_SIZE = 2048;
__constant__ float cBG[BG_SIZE];
__global__ void removeBG_constMemory(float* inBScan, float* outBScan, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int TileWidth){

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int Col = bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;
    uint idx = Row*BScanWidth + Col;

    // Calculate Ouptut Pixels
    if( (Row<BScanHeigth) && (Col<BScanWidth) )
        outBScan[idx] = satMin(inBScan[idx] - cBG[Col],0);

}
void removeBackground_constMemory(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    Stopwatch sw;
    CUDA_DeviceHelper devHelper(0,false);

    // Calc dimensions
    uint bWidth = 32*4;
    uint bHeight = ceil(1024/bWidth);
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bHeight);

    dim3 dimBlock(bWidth, bHeight, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    //devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cBG, bg_H, BG_SIZE*sizeof(float));

    // Execute Kernel
    removeBG_constMemory<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, width, height, bWidth);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Background constant Memory Kernel took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}

// Improved Kernel using Caching of Background
__global__ void removeBG_cachingBG(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int BlockWidth){


    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    // row and colums where Blocks will be taken verticaly to optimize caches
    // This works only in a square matrix (changing by an bx)
    uint colIdx = by*BlockWidth + tx;
    uint rowIdx = bx*BlockWidth + ty;
    uint idx = rowIdx*BScanWidth + colIdx;
    if( (rowIdx<BScanHeigth) && (colIdx<BScanWidth))
        outBScan[idx] = satMin(inBScan[idx] - bg[colIdx],0);
}
void removeBackground_cachingBG(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){

    CUDA_DeviceHelper devHelper(0, false);
    Stopwatch sw;

    // Calc dimensions
    uint bWidth = 32;
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bWidth); // /bWidth);
    if(gWidth > gHeight)
        gHeight = gWidth;
    else
        gWidth = gHeight;
    dim3 dimBlock(bWidth, bWidth, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    //devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *bG_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    int sizeBG = width*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &bG_D, sizeBG);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

    // Execute Kernel
    removeBG_cachingBG<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, bG_D, width, height, bWidth);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(bG_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Kernel using BG caching took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}

#if ARCH >= 60
    // Improved Kernel using Caching with dynamic parallelism (only Compute Capability 3.5 or higher allows dynamic parallelism)
    __global__ void removeBG_dynamicParallelism_child(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int startCol, unsigned int startRow){
        uint by = blockIdx.y;
        uint ty = threadIdx.y;

        uint col = startCol + threadIdx.x;
        uint row = startRow + by*blockDim.y + ty;
        uint idx = row*BScanWidth + col;
        if( (row<BScanHeigth) && (col<BScanWidth))
            outBScan[idx] = satMin(inBScan[idx] - bg[col],0);
    }
    __global__ void removeBG_dynamicParallelism_parent(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth, unsigned int BlockWidth){
        uint tx = threadIdx.x;

        // row and colums where Blocks will be taken verticaly to optimize caches
        // This works only in a square matrix (changing by an bx)

        dim3 dimGrid(1, ceil(BScanHeigth/(float)BlockWidth), 1);
        dim3 dimBlock(BlockWidth, BlockWidth, 1);

        removeBG_dynamicParallelism_child<<<dimGrid, dimBlock>>>(inBScan, outBScan, bg, BScanWidth, BScanHeigth, tx*BlockWidth, 0);

    }
    void removeBackground_dynamicParallelism(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){

        CUDA_DeviceHelper devHelper(0, false);
        Stopwatch sw;

        // Calc dimensions
        uint bWidth = 32;
        uint gWidth = ceil(width/bWidth);
        uint gHeight = ceil(height/bWidth);

        dim3 dimBlock(bWidth, 1, 1);
        dim3 dimGrid(1, 1, 1);
        //devHelper.checkDimConfigs(dimGrid, dimBlock);

        sw.Start();
        // Allocate Memory on GPU
        float *inBScan_D, *bG_D, *outBScan_D;
        int sizeBScan = width*height*sizeof(float);
        int sizeBG = width*sizeof(float);
        cudaMalloc((void**) &inBScan_D, sizeBScan);
        cudaMalloc((void**) &bG_D, sizeBG);
        cudaMalloc((void**) &outBScan_D, sizeBScan);

        // Copy Matrices to GPU memory
        cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
        cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

        // Execute Kernel
        removeBG_dynamicParallelism_parent<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, bG_D, width, height, bWidth);

        // Read back results from GPU memory
        cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

        // Free GPU memory
        cudaFree(inBScan_D);
        cudaFree(bG_D);
        cudaFree(outBScan_D);

        sw.Stop();
        cout << "Kernel using dynamic Parallelism: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
    }
#else
void removeBackground_dynamicParallelism(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    cout << "Kernel using dynamic Parallelism isn't available. GPU Arch: " << ARCH << ". If your Graphics card is newer, please change the constant in the kernel.cuh file." << endl;
}
#endif


__global__ void removeRowBG_constMemory(float* inBScan, float* outBScan, unsigned int BScanWidth, unsigned int BScanHeigth){
    uint col = blockIdx.x*blockDim.x+threadIdx.x;
    uint row = blockIdx.y;
    uint idx = row*BScanWidth + col;
    if( (row<BScanHeigth) && (col<BScanWidth))
        outBScan[idx] = satMin(inBScan[idx] - cBG[col],0);
}
void removeRowBackground_constMemory(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){
    Stopwatch sw;
    CUDA_DeviceHelper devHelper(0,false);

    // Calc dimensions
    uint bWidth = 32*4;
    uint gWidth = ceil(width/bWidth);
    uint gHeight = height;

    dim3 dimBlock(bWidth, 1, 1);
    dim3 dimGrid(gWidth, gHeight, 1);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cBG, bg_H, BG_SIZE*sizeof(float));

    // Execute Kernel
    removeRowBG_constMemory<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, width, height);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Background constant Memory Kernel took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}


// Improved Kernel using Caching of Background
__global__ void removeBG_optimBurst(float* inBScan, float* outBScan, float* bg, unsigned int BScanWidth, unsigned int BScanHeigth){

    // Calc Actual x, y Values of output element to calculate
    int tx = threadIdx.x;
    //int ty = threadIdx.y;
    int bx = blockIdx.x;
    int Row = blockIdx.y;
    int Col = bx * blockDim.x + tx;
    //int Row = by * blockDim.y + ty;
    //int Row = by;
    uint idx = Row*BScanWidth + Col;

    // Calculate Ouptut Pixels
    if( (Row<BScanHeigth) && (Col<BScanWidth) )
        outBScan[idx] = satMin(inBScan[idx] - bg[Col],0);
}
void removeBackground_optimBurst(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){

    CUDA_DeviceHelper devHelper(0, false);
    Stopwatch sw;

    // Calc dimensions
    uint bWidth = 32*4;
    uint bHeight = 1;
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bHeight); // /bWidth);

    dim3 dimBlock(bWidth, bHeight, 1);
    dim3 dimGrid(gWidth, gHeight, 1);
    devHelper.checkDimConfigs(dimGrid, dimBlock);

    sw.Start();
    // Allocate Memory on GPU
    float *inBScan_D, *bG_D, *outBScan_D;
    int sizeBScan = width*height*sizeof(float);
    int sizeBG = width*sizeof(float);
    cudaMalloc((void**) &inBScan_D, sizeBScan);
    cudaMalloc((void**) &bG_D, sizeBG);
    cudaMalloc((void**) &outBScan_D, sizeBScan);

    // Copy Matrices to GPU memory
    cudaMemcpy(inBScan_D, inBScan_H, sizeBScan, H2D); //cudaMemcpyHostToDevice);
    cudaMemcpy(bG_D, bg_H, sizeBG, H2D);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); // Enlarge L1 cache size in favour of Shared Memory

    // Execute Kernel
    removeBG_optimBurst<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, bG_D, width, height);

    // Read back results from GPU memory
    cudaMemcpy(outBScan_H, outBScan_D, sizeBScan, D2H);

    // Free GPU memory
    cudaFree(inBScan_D);
    cudaFree(bG_D);
    cudaFree(outBScan_D);

    sw.Stop();
    cout << "Kernel using BG caching and optimal Burst sizes: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}
