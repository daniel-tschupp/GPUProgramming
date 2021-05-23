#include "kernels.cuh"

using namespace std;

__global__ void removeBG_cachingBG(BackgroundKernelParams params){
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    // row and colums where Blocks will be taken verticaly to optimize caches
    // This works only in a square matrix (changing by an bx)
    uint colIdx = by*params.blockWidth + tx;
    uint rowIdx = bx*params.blockWidth + ty;
    uint idx = rowIdx*params.bScanWidth + colIdx;
    if( (rowIdx<params.bScanHeight) && (colIdx<params.bScanWidth))
        params.dBScanOut[idx] = satMin(params.dBScanIn[idx] - params.dBackground[colIdx],0);
}


void executeBGRemoveKernel(cudaStream_t* stream, BackgroundKernelParams params){
    uint bWidth = params.blockWidth;
    uint gWidth = ceil(params.bScanWidth/bWidth);
    uint gHeight = ceil(params.bScanHeight/bWidth);
    if(gWidth > gHeight)
        gHeight = gWidth;
    else
        gWidth = gHeight;
    dim3 dimBlock(bWidth, bWidth, 1);
    dim3 dimGrid(gWidth, gHeight, 1);

    removeBG_cachingBG<<<dimGrid, dimBlock, 0, *stream>>>(params);
}
