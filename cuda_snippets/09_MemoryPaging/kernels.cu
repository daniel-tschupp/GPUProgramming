#include "kernels.cuh"

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
    uint gHeight = ceil(height/bWidth);
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

void removeBackground_cachingBG_Mapped(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height){

    CUDA_DeviceHelper devHelper(0, false);
    Stopwatch sw;

    // Calc dimensions
    uint bWidth = 32;
    uint gWidth = ceil(width/bWidth);
    uint gHeight = ceil(height/bWidth);
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


    cudaHostGetDevicePointer(&inBScan_D, inBScan_H, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&outBScan_D, outBScan_H, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&bG_D, bg_H, cudaHostAllocMapped);

    // Execute Kernel
    removeBG_cachingBG<<<dimGrid, dimBlock>>>(inBScan_D, outBScan_D, bG_D, width, height, bWidth);

    cudaDeviceSynchronize();

    sw.Stop();
    cout << "Kernel using BG caching took: " << sw.GetElapsedTimeMilliseconds() << "ms. " << endl;
}
