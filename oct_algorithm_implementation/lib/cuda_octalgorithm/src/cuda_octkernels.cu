/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_octkernels.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

struct HW_Signal{
    unsigned int tx, ty, bx, by, bDimx, bDimy, colIn, colOut, row, globIdxIn, globIdxOut, smIdx, smWidth, smHeight, smCol;
};
struct ComplexNumber {
   union { // anonymous union
      cufftComplex cufftComplexValue;       // cufftComplex value
      struct { float x, y; };               // anonymous structure to access x, y of cufftComplex
      struct { float real, imag; };         // anonymous structure to access real and imaginary part of cufftComplex
      struct { float amp, phi; };
   };
};
__device__ inline void __checkIdx(unsigned int idx, unsigned int min, unsigned int max, HW_Signal* sig, const char *file, const int line){
#ifdef CUDA_DEBUG_MSG
    if(idx<min || idx>max){
        printf("Kernel failed at: %s:%i : Block: %d, Thread: %d, Idx = %d\n", file, line, sig->bx, sig->tx, idx);
    }
#endif
}
#define CHECK_IDX(idx, min, max, sig) __checkIdx(idx, min, max, sig, __FILE__, __LINE__);

#define IF_TH1  if( threadIdx.x==0 && threadIdx.y==0 && blockIdx.y<=1 )

/* Defining Constant Memory Variables **********************************************************************************************************************/
__constant__ unsigned int constSizeSMStartInds;
__constant__ unsigned int constWindowPmin;
__constant__ unsigned int constWindowPmax;
__constant__ float constAlpha;
__constant__ unsigned int constProcessingOptions;
__constant__ unsigned int smStartInds[128];
__constant__ float constDispCoeff[3];


/***********************************************************************************************************************************************************/
/*******************************************************************       local        *******************************************************************/
/**********************************************************************************************************************************************************/

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              1       /   0
 *
 * Operations Done: 0
 * ******************************************************************/
__device__ void convCpyToShared(float* sharedMem, unsigned short* globMem, HW_Signal* sig, unsigned int padLeft, unsigned int padRight, unsigned int colBound, unsigned int rowBound){
    if(sig->colIn < colBound && sig->row < rowBound){
        sharedMem[sig->smIdx] = globMem[sig->globIdxIn];
        if(sig->tx < padLeft){
            if((int)sig->colIn-(int)padLeft >= 0)
                sharedMem[sig->smIdx-padLeft] = globMem[sig->globIdxIn-padLeft];
            else
                sharedMem[sig->smIdx-padLeft] = 0;
        }
        if(sig->tx >= sig->bDimx-padRight){
            if(sig->colIn+padRight < colBound)
                sharedMem[sig->smIdx+padRight] = globMem[sig->globIdxIn+padRight];
            else
                sharedMem[sig->smIdx+padRight] = 0;
        }
    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 16
 * ******************************************************************/
__device__ void changeEndian(float* inout, HW_Signal* sig, unsigned int padLeft, unsigned int padRight, unsigned int colBound, unsigned int rowBound){
    if(sig->colIn < colBound && sig->row < rowBound){
        if(IS_CHANGE_ENDIAN(constProcessingOptions)){
            unsigned short minor = (unsigned short)((inout[sig->smIdx]+0.1f)) & 0x00FF;
            unsigned short major = (unsigned short)((inout[sig->smIdx]+0.1f)) & 0xFF00;
            inout[sig->smIdx] = (float)((minor << 8) | (major >> 8));

            if(sig->tx < padLeft){
                if((int)sig->colIn-(int)padLeft >= 0){
                    minor = (unsigned short)((inout[sig->smIdx-padLeft]+0.1f)) & 0x00FF;
                    major = (unsigned short)((inout[sig->smIdx-padLeft]+0.1f)) & 0xFF00;
                    inout[sig->smIdx-padLeft] = (float)((minor << 8) | (major >> 8));
                }
            }
            if(sig->tx >= sig->bDimx-padRight){
                if(sig->colIn+padRight < colBound){
                    minor = (unsigned short)((inout[sig->smIdx+padRight]+0.1f)) & 0x00FF;
                    major = (unsigned short)((inout[sig->smIdx+padRight]+0.1f)) & 0xFF00;
                    inout[sig->smIdx+padRight] = (float)((minor << 8) | (major >> 8));
                }
            }


        }
    }
}


/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              1       /   0
 *
 * Operations Done: 5 / 6
 * ******************************************************************/
__device__ void removeBackground(float* inout, OctGPUParams* params, HW_Signal* sig, unsigned int colBound, unsigned int rowBound){
    if(sig->colIn < colBound && sig->row < rowBound){
        if(IS_BGREMOVE_ACTIVE(constProcessingOptions)){
            inout[sig->smIdx] -= params->baGround_C[sig->colIn];
            if(sig->tx < params->padLeft){
                if((int)sig->colIn-(int)params->padLeft >= 0 && (int)sig->smIdx-(int)params->padLeft >= 0)
                    inout[sig->smIdx-params->padLeft] -= params->baGround_C[sig->colIn-params->padLeft];
            }
            if(sig->tx >= sig->bDimx-params->padRight){
                if(sig->colIn+params->padRight < colBound)
                    inout[sig->smIdx+params->padRight] -= params->baGround_C[sig->colIn-params->padRight];
            }
        }
    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 29
 * ******************************************************************/
__device__ void movingAvg(float* out, float* in, unsigned int smCol, OctGPUParams* params,HW_Signal* sig){
    float res = 0;
    float cnt = 0;
    int smIdx = smCol + sig->ty*sig->smWidth;
    for( int i = -((int)params->dcKernelSize/2); i < ((int)params->dcKernelSize/2); ++i){
        if(smCol + i > 0 && smCol + i < (int)sig->smWidth){
            res += in[(unsigned int)smIdx+i];
            cnt++;
        }
    }
    out[smIdx] = in[smIdx] - (res/cnt);
}
__device__ void movingAverage(float* out, float* in, OctGPUParams* params, HW_Signal* sig, unsigned int colBound, unsigned int rowBound){
    if(sig->colIn < colBound && sig->row < rowBound){
        if(IS_DCREMOVE_ACTIVE(constProcessingOptions)){
            movingAvg(out, in, sig->smCol, params, sig);

            if(sig->tx < params->padLeft){
                movingAvg(out, in, sig->smCol-params->padLeft, params, sig);
            }
            if(sig->tx >= sig->bDimx-params->padRight){
                movingAvg(out, in, sig->smCol+params->padRight, params, sig);
            }
        }
        else{
            out[sig->smIdx] = in[sig->smIdx];

            if(sig->tx < params->padLeft){
                out[sig->smIdx-params->padLeft] = in[sig->smIdx-params->padLeft];
            }
            if(sig->tx >= sig->bDimx-params->padRight){
                out[sig->smIdx+params->padRight] = in[sig->smIdx+params->padRight];
            }
        }

    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 2
 * ******************************************************************/
__device__ void spectralShaping(float* inout, float* background, OctGPUParams* params, struct HW_Signal* sig, unsigned int colBound, unsigned int rowBound){
    if(sig->colIn < colBound && sig->row < rowBound){
        if(IS_SPECTRALSHAPPING_ACTIVE(constProcessingOptions)){ // ld/st: 2 op: 1
            inout[sig->smIdx] *= params->bgMean / background[sig->colIn];
        }
    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              2       /   0
 *
 * Operations Done: 11
 * ******************************************************************/
__device__ void convRemapKernel(float* out, float* in, OctGPUParams* params, HW_Signal* sig, unsigned int dataStartCol, unsigned int colBound, unsigned int rowBound){
    if(sig->colOut < colBound && sig->row < rowBound){
        if(IS_REMAP_ACTIVE(constProcessingOptions)){
            float res = 0;

            for(unsigned int iConvKernel = 0; iConvKernel < params->kernelSize; ++iConvKernel){
                unsigned int idx = sig->colOut*params->kernelSize+iConvKernel;
                CHECK_IDX(idx, 0, (unsigned int)params->nbrOfSamplesAScanRaw*params->overSamplingRatio*params->kernelSize-1, sig);

                if(params->convInds_C[idx] != -1){
                    CHECK_IDX(params->convInds_C[idx], 0, params->nbrOfSamplesAScanRaw, sig);
                    unsigned int smInIdx = sig->ty*sig->smWidth+(unsigned int)(params->convInds_C[idx] + (int)params->padLeft - dataStartCol);
                    CHECK_IDX(smInIdx, 0, sig->smWidth*sig->smHeight, sig);
                    res += in[smInIdx] * params->convCoeff_C[idx];
                }
            }

            CHECK_IDX(sig->smIdx, 0, sig->bDimy*(sig->bDimx+params->padLeft+params->padRight+params->kernelSize)-1, sig);
            out[sig->smIdx] = res;
        }
        else {
            out[sig->smIdx] = in[sig->smIdx];
        }
    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 9
 * ******************************************************************/
#define CUDART_PI_F 3.141592654f

__device__ float calcWindow(unsigned int actualIndex){
    unsigned int pmin = constWindowPmin;
    unsigned int pmax = constWindowPmax;
    if(IS_REMAP_ACTIVE(constProcessingOptions)){
        pmin = (unsigned int)ceilf(pmin * constAlpha);
        pmax = (unsigned int)floorf(pmax * constAlpha);
    }
    unsigned int nbrOfSamplesAScanCropped = (unsigned int)(pmax - pmin + 1);
    if(actualIndex >= pmin && actualIndex < nbrOfSamplesAScanCropped+pmin ){
        unsigned int i = actualIndex - pmin;
        return 0.5f*((sinf((2.0f*CUDART_PI_F/(nbrOfSamplesAScanCropped-1))*i +  1.5f*CUDART_PI_F))+1);
    }
    return 0.0f;
}
/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 10
 * ******************************************************************/
__device__ void applyWindow(float* inout, HW_Signal* sig, unsigned int colBound, unsigned int rowBound){
    if(sig->colOut < colBound && sig->row < rowBound) {
        if(IS_WINDOWING_ACTIVE(constProcessingOptions)){ // ld/st: 2 op: 1
                inout[sig->smIdx] *= calcWindow(sig->colOut);
        }
    }
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              0       /   0
 *
 * Operations Done: 14
 * ******************************************************************/
//__constant__ float constDispCoeff[3];
__device__ ComplexNumber calcDispersion(unsigned int actualIndex, unsigned int nbrOfSamples){
    ComplexNumber disp;
    float coeff = 1.0f / (nbrOfSamples-1.0f);

    // a2-a4  set by user
    float a2 = constDispCoeff[0]; float a3 =  constDispCoeff[1]; float a4 = constDispCoeff[2];

    float x = -0.5f + coeff*actualIndex;
    x *= x;         // x^2
    float phi = a2*x;
    x *= x;         // x^3
    phi += a3*x;
    x *= x;         // x^4
    phi += a4*x;

    disp.real = cosf(phi);
    disp.imag = sinf(phi);

    return disp;
}

__device__ void applyDispersionCompensation(cufftComplex* out, float* in, HW_Signal* sig, unsigned int colBound, unsigned int rowBound){
    if(sig->colOut < colBound && sig->row < rowBound)// ld/st: 2 op 16
    {
        if(IS_DISPCOMP_ACTIVE(constProcessingOptions)){
            ComplexNumber disp = calcDispersion(sig->colOut, colBound);
            out[sig->globIdxOut].x = in[sig->smIdx] * disp.real;
            out[sig->globIdxOut].y = in[sig->smIdx] * disp.imag;
        }
        else{
            out[sig->globIdxOut].x = in[sig->smIdx];
            out[sig->globIdxOut].y = in[sig->smIdx];
        }
    }
}

/***********************************************************************************************************************************************************/
/*******************************************************************       global        *******************************************************************/
/**********************************************************************************************************************************************************/
    __global__ void preFFTProcess(OctGPUParams params, OctIOdata ioData)
    {
    // Allocate gpu hardware indexing singals
    struct HW_Signal sig {threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // The data partitioning for this kernel follows the output data partitioning scheme. As the input datas are shorter than
    // the output data and the remap isn't linear too, it is crutial to select the proper portion of the input datas to be
    // able to calculate the output datas.
    // While creating the remap vector, the starting data position for all tiles are calculated too and stored to the constant
    // memory. This is used to calculate the actual input data column and input data index.
    // This section of the kernel code defines input data pointers, output data pointers as well as the shared memory pointers.
    sig.row = sig.by * sig.bDimy + sig.ty;

    unsigned int lineLength = params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor;
    unsigned int colBoundIn = params.nbrOfSamplesAScanRaw;
    unsigned int colBoundOut = colBoundIn;
    unsigned int rowBound = params.nbrOfAScans;
    unsigned int dataStartCol;
    if(IS_REMAP_ACTIVE(constProcessingOptions)){
        lineLength *= params.overSamplingRatio;
        colBoundOut *= params.overSamplingRatio;
        dataStartCol = smStartInds[sig.bx];
    }
    else{
        dataStartCol = (unsigned int)((float)sig.bx*(float)sig.bDimx);
    }
    sig.colIn = dataStartCol + sig.tx;
    sig.globIdxIn = sig.row*params.nbrOfSamplesAScanRaw + sig.colIn;

    sig.colOut = sig.bx*sig.bDimx + sig.tx;
    sig.globIdxOut = sig.row*lineLength + sig.colOut;

    sig.smWidth = sig.bDimx + params.padLeft + params.padRight;
    sig.smHeight = sig.bDimy;
    sig.smCol = sig.tx + params.padLeft;
    sig.smIdx = sig.ty*sig.smWidth + sig.smCol;

    // Allocate shared memory resources for the tiles
    extern __shared__ char sharedMem[];
    float* bScan1 = (float*)sharedMem;
    float* bScan2 = (float*)(sharedMem + sig.smWidth*sig.smHeight*sizeof(float));

    // Copy data of tile from global memory to shared memory
    //  Data alingemn:
    //  aScan1: 0....n
    //  aScan2: 0....n
    //  aScan3: 0....n

#ifdef DEBUG_KERNEL_BOUNDARIES
    IF_TH1{
        printf("----------------Kernel-------- BlockDim x: %d\n", blockDim.x);
        printf("----------------Kernel-------- BlockDim y: %d\n", blockDim.y);
        printf("----------------Kernel-------- lineLength: %d\n", lineLength);
        printf("----------------Kernel-------- lineLength: %d\n", lineLength);
        printf("----------------Kernel-------- lineLength: %d\n", lineLength);
        printf("----------------Kernel-------- smWidth: %d\n", sig.smWidth);
        printf("----------------Kernel-------- smHeight: %d\n", sig.smHeight);
        printf("----------------Kernel-------- globIdxIn: %d\n", sig.globIdxIn);
        printf("----------------Kernel-------- globIdxOut: %d\n", sig.globIdxOut);
        printf("----------------Kernel-------- dataStartCol: %d\n", dataStartCol);
        printf("----------------Kernel-------- colBoundIn: %d\n", colBoundIn);
        printf("----------------Kernel-------- colBoundOut: %d\n", colBoundOut);
        printf("----------------Kernel-------- rowBound: %d\n", rowBound);
    }
#endif
    convCpyToShared(bScan1, ioData.bScanRaw_C, &sig, params.padLeft, params.padRight, colBoundIn, rowBound); // ld/st:2  op:0
    // ******************** Executing Kernel task ********************************************

    // Reversing Bits ************************************************************************
    changeEndian(bScan1, &sig, params.padLeft, params.padRight, colBoundIn, rowBound);      // ld/st:0 op:16

    // Remove BG *****************************************************************************
    removeBackground(bScan1, &params, &sig, colBoundIn, rowBound);                                  // ld/st:1  op:5
    __syncthreads();
    // Remove DC *****************************************************************************
    movingAverage(bScan2, bScan1, &params, &sig, colBoundIn, rowBound);                            // ld/st:0  op:29

    // Spectral Shaping **********************************************************************
    spectralShaping(bScan2, params.baGround_C, &params, &sig, colBoundIn, rowBound);                 // ld/st:0 op 2
    __syncthreads();

    convRemapKernel(bScan1, bScan2, &params, &sig, dataStartCol, colBoundOut, rowBound);            // ld/st:2  op:11

    // windowing *****************************************************************************
    applyWindow(bScan1, &sig, colBoundOut, rowBound);                                      //  ld/st:0 op 10

    // Copy result back to global memory and add dispersion **********************************
    if(IS_REMAP_ACTIVE(constProcessingOptions)){
        applyDispersionCompensation(ioData.bScanPreFFTResultRemap_C, bScan1, &sig, colBoundOut, rowBound);
    }
    else {
        applyDispersionCompensation(ioData.bScanPreFFTResult_C, bScan1, &sig, colBoundOut, rowBound);
    }
    // Compute to global load/store ratio: 83 / 7 = 11.9    (perfect is (GFLOPS/2) / MemoryBandwidth
    // Ideal C-Ratio for K420:      336GFLOPS/s/2   / 29GB/s    = 5.8
    // Ideal C-Ratio for P1000:     1894GFLOPS/s/2  / 80GB/s    = 11.85
    // Ideal C-Ratio for 1080Ti:    11340GFLOPS/s/2 / 484GB/S   = 11.7
}

/*********************************************************************
 *
 * Gobal Memory Loads   /   Stores per Thread
 *              3       /   1
 *
 * Operations Done: 7
 * ******************************************************************/
__global__ void postFFTProcess(OctGPUParams params, OctIOdata ioData, float scaling){
    struct HW_Signal sig {threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    sig.colOut = blockIdx.x * blockDim.x + threadIdx.x;
    sig.row = blockIdx.y * blockDim.y + threadIdx.y;

    if(IS_REMAP_ACTIVE(constProcessingOptions)){
        unsigned int nbrOfSampleAScanProcessing = (unsigned int)(params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor*params.overSamplingRatio);
        sig.globIdxOut = sig.row*nbrOfSampleAScanProcessing + sig.colOut;
        if(sig.colOut < nbrOfSampleAScanProcessing/2 && sig.row < params.nbrOfAScans)
        {
            if(sig.colOut >= 0){
                float valX = ioData.bScanCom_C[sig.globIdxOut].x * params.conCn_C[sig.colOut];
                float valY = ioData.bScanCom_C[sig.globIdxOut].y * params.conCn_C[sig.colOut];
                ioData.bScanRes_C[sig.globIdxOut + nbrOfSampleAScanProcessing/2]= scaling * sqrtf( valX*valX + valY*valY );
            }
        }
        else if(sig.colOut >= nbrOfSampleAScanProcessing/2 && sig.colOut < nbrOfSampleAScanProcessing && sig.row < params.nbrOfAScans)
        {
            if(sig.colOut < nbrOfSampleAScanProcessing){
                float valX = ioData.bScanCom_C[sig.globIdxOut].x * params.conCn_C[sig.colOut];
                float valY = ioData.bScanCom_C[sig.globIdxOut].y * params.conCn_C[sig.colOut];
                ioData.bScanRes_C[sig.globIdxOut - nbrOfSampleAScanProcessing/2]= scaling * sqrtf( valX*valX + valY*valY );
            }
        }
    }
    else{
        unsigned int nbrOfSampleAScanProcessing = (unsigned int)(params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor);
        sig.globIdxOut = sig.row*nbrOfSampleAScanProcessing + sig.colOut;
        if(sig.colOut < nbrOfSampleAScanProcessing/2 && sig.row < params.nbrOfAScans)
        {
            if(sig.colOut >= 0){
                float valX = ioData.bScanCom_C[sig.globIdxOut].x;
                float valY = ioData.bScanCom_C[sig.globIdxOut].y;
                ioData.bScanRes_C[sig.globIdxOut + nbrOfSampleAScanProcessing/2]= scaling * sqrtf( valX*valX + valY*valY );
            }
        }
        else if(sig.colOut >= nbrOfSampleAScanProcessing/2 && sig.colOut < nbrOfSampleAScanProcessing && sig.row < params.nbrOfAScans)
        {
            if(sig.colOut < nbrOfSampleAScanProcessing){
                float valX = ioData.bScanCom_C[sig.globIdxOut].x;
                float valY = ioData.bScanCom_C[sig.globIdxOut].y;
                ioData.bScanRes_C[sig.globIdxOut - nbrOfSampleAScanProcessing/2]= scaling * sqrtf( valX*valX + valY*valY );
            }
        }
    }
}

__global__ void byPassFFT(OctGPUParams params, OctIOdata ioData, float scaling)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(IS_REMAP_ACTIVE(constProcessingOptions)){
        unsigned int index = (unsigned int)(i + j * params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor);
        if( i < params.nbrOfSamplesAScanRaw*params.overSamplingRatio && j < params.nbrOfAScans)
        {
            ioData.bScanRes_C[index] =  scaling * ioData.bScanPreFFTResultRemap_C[index].x;
        }
        else
        {
            if(i < params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor && j < params.nbrOfAScans)
                ioData.bScanRes_C[index] = 0;
        }
    }
    else{
        unsigned int index = (unsigned int)(i + j * params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor);
        if( i < params.nbrOfSamplesAScanRaw && j < params.nbrOfAScans)
        {
            ioData.bScanRes_C[index] =  scaling * ioData.bScanPreFFTResult_C[index].x;
        }
        else
        {
            if(i < params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor && j < params.nbrOfAScans)
                ioData.bScanRes_C[index] = 0;
        }
    }
}

#ifdef __cplusplus
}
#endif

/* Copyright details:
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
**
** Contact: Daniel Tschupp ( daniel.tschupp@gmail.com )
*/

