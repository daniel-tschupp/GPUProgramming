/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
**
** Cro = cropped
** Com = complex
** Con = convolution
** Org = orginal
** Raw = raw
** Res = result
**
** baGround  = Back Ground vector
** conNIter  = convolution N iterations
** conRemVe  = convolution remap vector
** conKernC  = convolution kernel
** DispAWind = dispersion compensation and windowing
** fftHandle_forDC_C = fft plan with A-Scan length of the origin
** fftHandleResult_C = fft plan with upsampled A-Scan length
**
*/
#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_processoctdata.h"
#include "cuda_octkernels.h"
#include "cuda_utils.h"
#include "cuda_errorchecking.h"
#include <assert.h>



/***********************************************************************************************************************************************************/
/*******************************************************************     Process GPU     *******************************************************************/
void ProcessGPU(OctGPUParams params, OctIOdata ioData, int processingOptions)
{
    assert(ioData.stream != nullptr);
    assert(ioData.srcBscan != nullptr);
    assert(ioData.dstResult != nullptr);
#ifdef DEBUG_GLOB_ALLOC_MEM
    printf("Stream address: %d\n", (unsigned long int)ioData.stream);
#endif
    gpuWaitForStream(ioData.stream); // waits here for the actual stream to be ready.

    //--copy Bscan from Host to Device
    int sizeBscanRaw = params.nbrOfSamplesAScanRaw * params.nbrOfAScans * sizeof(unsigned short);

    // Copy raw-data from cpu to gpu
    gpuCpyAsyncToDevice(ioData.bScanRaw_C, ioData.srcBscan, sizeBscanRaw, ioData.stream, false);

    // Main Processing
    params.dcKernelSize = 25;
    params.padLeft = params.dcKernelSize/2;
    params.padRight = params.tileWidth*0.25;
    unsigned int TileWidth = params.tileWidth;
    unsigned int TileHeight = 1;
    unsigned int smSize = 2 * (TileWidth+params.padLeft+params.padRight)*TileHeight * sizeof(float);

    dim3 dimBlock(TileWidth, TileHeight, 1);
    unsigned int blocksX, blockXPostFFT;
    if(IS_REMAP_ACTIVE(processingOptions)){
        blocksX = (int)ceilf((float)params.nbrOfSamplesAScanRaw*params.overSamplingRatio/(float)TileWidth-0.00001f);
        blockXPostFFT = (int)ceilf((float)params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor/(float)TileWidth-0.00001f);
    }
    else{
        blocksX = (int)ceilf((float)params.nbrOfSamplesAScanRaw/(float)TileWidth-0.00001f);
        blockXPostFFT = (int)ceilf((float)params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor/(float)TileWidth-0.00001f);
    }

    unsigned int blocksY = (int)ceilf((float)params.nbrOfAScans/(float)TileHeight);
    dim3 dimGrid(blocksX, blocksY, 1);
    dim3 dimGridPostFFT(blockXPostFFT, blocksY, 1);

#ifdef DEBUG_KERNEL_SIZE
    printf("Kernel size Infos PreFFT:\n");
    printf("Grid Size: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Block Size: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("Shared Memory Size: %d\n", smSize);
    printf("Kernel size Infos PostFFT:\n");
    printf("Grid Size: %d x %d x %d\n", dimGridPostFFT.x, dimGridPostFFT.y, dimGridPostFFT.z);
    printf("Block Size: %d x %d x %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z);
#endif
    preFFTProcess<<<dimGrid,dimBlock,smSize,*ioData.stream>>>( params , ioData);
    CudaCheckKernelError();

    if(IS_FFT_ACTIVE(processingOptions))
    {
        // IFFT
        if(IS_REMAP_ACTIVE(processingOptions)){
            cufftSetStream(ioData.fftHandleResultRemap_C, *ioData.stream);
            CudaSafeFFTCall(cufftExecC2C(ioData.fftHandleResultRemap_C, ioData.bScanPreFFTResultRemap_C, ioData.bScanCom_C, CUFFT_INVERSE));
        }
        else{
            cufftSetStream(ioData.fftHandleResult_C, *ioData.stream);
            CudaSafeFFTCall(cufftExecC2C(ioData.fftHandleResult_C, ioData.bScanPreFFTResult_C, ioData.bScanCom_C, CUFFT_INVERSE));
        }
        postFFTProcess<<<dimGridPostFFT, dimBlock, 0, *ioData.stream>>>(params, ioData, 1.0f/(float)(params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor));
        CudaCheckKernelError();
    }
    else
    {
        byPassFFT<<<dimGridPostFFT,dimBlock,0,*ioData.stream>>>(params, ioData, 1.0f);
        CudaCheckKernelError();
    }

    int sizeBscanRes = params.nbrOfSamplesAScanRaw * params.zeroPaddingFactor * params.nbrOfAScans * sizeof(float);
    if(IS_REMAP_ACTIVE(processingOptions))
        sizeBscanRes *= params.overSamplingRatio;

#ifdef DEBUG_OPTIONS
    printf("Options: \n");
    printf("Over sampling ratio: %g\n", params.overSamplingRatio);
    printf("Zero padding factor: %g\n", params.zeroPaddingFactor);
    printf("Nbr of Samples in A Scans Raw: %d\n", params.nbrOfSamplesAScanRaw);
    printf("Nbr of Samples in A Scans Result: %d\n", (int)(sizeBscanRes/(params.nbrOfAScans * sizeof(float))));
    printf("Nbr A Scans in B Scan: %d\n", params.nbrOfAScans);
    if(IS_REMAP_ACTIVE(processingOptions))
        printf("REMAP active\n");
    else
        printf("REMAP not active\n");

    if(IS_FFT_ACTIVE(processingOptions))
        printf("FFT active\n");
    else
        printf("FFT not active\n");

    if(IS_DCREMOVE_ACTIVE(processingOptions))
        printf("DC REMOVE active\n");
    else
        printf("DC REMOVE not active\n\n");
#endif

    gpuCpyAsyncToHost(ioData.bScanRes_C, ioData.dstResult, sizeBscanRes, ioData.stream, false);
}

/******************************************************************* Init Cuda Buffer GPU *******************************************************************/
void InitConfigCudaBuffers(OctGPUParams* params)
{
    // Set sizes
    int sizeConCnC   = params->nbrOfSamplesAScanRaw*params->overSamplingRatio*params->zeroPaddingFactor * sizeof(float);
    int sizeBaGround = params->nbrOfSamplesAScanRaw * sizeof(float);
    int sizeConvCoef = params->nbrOfSamplesAScanRaw*params->overSamplingRatio * params->kernelSize * sizeof(float);
    int sizeConvInds = params->nbrOfSamplesAScanRaw*params->overSamplingRatio * params->kernelSize * sizeof(int);

    // Cuda Malloc
    params->conCn_C = (float*)gpuMallocDeviceMemory(sizeConCnC);
    params->baGround_C = (float*)gpuMallocDeviceMemory(sizeBaGround);
    params->convCoeff_C = (float*)gpuMallocDeviceMemory(sizeConvCoef);
    params->convInds_C= (int*)gpuMallocDeviceMemory(sizeConvInds);

    bool PrintGPUInfos=false;
#ifdef DEBUG_GPU_SPECS
    PrintGPUInfos=true;
#endif
    if(PrintGPUInfos) gpuGetGPUInfos();
}

void InitDataCudaBuffers(OctIOdata* ioData, const OctGPUParams params)
{
    int sizeBscanRes = params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor * params.nbrOfAScans * sizeof(float);
    int sizeBscanRaw = params.nbrOfSamplesAScanRaw * params.nbrOfAScans * sizeof(unsigned short);
    int sizeBscanCom = params.nbrOfSamplesAScanRaw*params.overSamplingRatio*params.zeroPaddingFactor * params.nbrOfAScans * sizeof(cufftComplex);

    ioData->bScanRes_C = (float*)gpuMallocDeviceMemory(sizeBscanRes);
    ioData->bScanRaw_C = (unsigned short*)gpuMallocDeviceMemory(sizeBscanRaw);
    ioData->bScanCom_C = (float2*)gpuMallocDeviceMemory(sizeBscanCom);
    ioData->bScanPreFFTResult_C = (float2*)gpuMallocDeviceMemory(sizeBscanCom);
    ioData->bScanPreFFTResultRemap_C = (float2*)gpuMallocDeviceMemory(sizeBscanCom);

    CudaSafeFFTCall(cufftPlan1d(&ioData->fftHandleResult_C, params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor, CUFFT_C2C, params.nbrOfAScans ));
    CudaSafeFFTCall(cufftPlan1d(&ioData->fftHandleResultRemap_C, params.nbrOfSamplesAScanRaw*params.zeroPaddingFactor*params.overSamplingRatio, CUFFT_C2C, params.nbrOfAScans ));
}

/******************************************************************* Free Cuda Buffer GPU *******************************************************************/

void FreeCudaConfigBuffers(OctGPUParams* params)
{
    gpuFreeDeviceMemory( params->conCn_C);
    gpuFreeDeviceMemory( params->baGround_C);
    gpuFreeDeviceMemory( params->convCoeff_C);
    gpuFreeDeviceMemory( params->convInds_C);
}
void FreeCudaDataBuffers(OctIOdata* ioData){
    gpuFreeDeviceMemory( ioData->bScanRaw_C );
    gpuFreeDeviceMemory( ioData->bScanRes_C );
    gpuFreeDeviceMemory( ioData->bScanCom_C );
    gpuFreeDeviceMemory( ioData->bScanPreFFTResult_C);
    gpuFreeDeviceMemory( ioData->bScanPreFFTResultRemap_C);

    cufftDestroy( ioData->fftHandleResult_C );
    cufftDestroy( ioData->fftHandleResultRemap_C );
}

/********************************************************************* Constant Memory Pointer Handling ****************************************************/
extern __constant__ unsigned int constSizeSMStartInds;
extern __constant__ unsigned int constWindowPmin;
extern __constant__ unsigned int constWindowPmax;
extern __constant__ float constAlpha;
extern __constant__ unsigned int constProcessingOptions;
extern __constant__ unsigned int smStartInds[];
extern __constant__ float constDispCoeff[];

void writeCudaConstSizeSMStartInds(int* sizeSMStartInds){
    assert(sizeSMStartInds != nullptr);
    if(*sizeSMStartInds > 128)
        printf("WARING: To many Elements in SM Start Inds!");
    CudaSafeAPICall(cudaMemcpyToSymbol(constSizeSMStartInds, sizeSMStartInds, sizeof(unsigned int)));
}
void writeCudaConstSMStartInds(unsigned int* startInds, unsigned int elements){
#ifdef DEBUG_STARTINDS
    printf("Printing Start Indeces written to graphics card:\n");
    for(int i = 0; i<elements; ++i)
        printf("Element %d: %d\n", i, startInds[i]);
#endif
    CudaSafeAPICall(cudaMemcpyToSymbol(smStartInds, startInds, elements*sizeof(unsigned int)));
}
void writeCudaConstDispCoeff(float* dispCoef, unsigned int elements){
    assert(dispCoef != nullptr);
    CudaSafeAPICall(cudaMemcpyToSymbol(constDispCoeff, dispCoef, elements*sizeof(float)));
}
void writeCudaConstWindowPmin(int* windowPmin){
    assert(windowPmin != nullptr);
    CudaSafeAPICall(cudaMemcpyToSymbol(constWindowPmin, windowPmin, sizeof(unsigned int)));
}
void writeCudaConstWindowPmax(int* windowPmax){
    assert(windowPmax != nullptr);
    CudaSafeAPICall(cudaMemcpyToSymbol(constWindowPmax, windowPmax, sizeof(unsigned int)));
}
void writeCudaConstProcessingOptions(int* proc_options){
    assert(proc_options != nullptr);
    CudaSafeAPICall(cudaMemcpyToSymbol(constProcessingOptions, proc_options, sizeof(unsigned int)));
}
void writeCudaConstOverSamplingRatio(float* overSamplingRatio){
    assert(overSamplingRatio != nullptr);
    CudaSafeAPICall(cudaMemcpyToSymbol(constAlpha, overSamplingRatio, sizeof(float)));
}
void writeCudaBuffer(void* gpuBuffer, void* hostBuffer, unsigned int nbrOfBytes){
    assert(gpuBuffer != nullptr);
    assert(hostBuffer != nullptr);
    gpuCpySyncToDevice(gpuBuffer, hostBuffer, nbrOfBytes);
}

/************************************************* Labview Wrapper Functions ****************************************************/

OctIOdata* createNewStreamPtrLabView(OctIOdata* ioData){
    ioData->stream = gpuGetNewStream();
    return ioData;
}
OctIOdata*  cleanUpStreamLabView(OctIOdata* ioData){
    gpuCleanUpStream(ioData->stream);
    return ioData;
}

OctIOdata* ProcessGPULabview(OctGPUParams* params, OctIOdata* ioData, int processingOptions, unsigned short srcBScan[], float dstResult[])
{
    ioData->srcBscan = srcBScan;
    ioData->dstResult = dstResult;
    ProcessGPU(*params, *ioData, processingOptions);
    gpuWaitForStream(ioData->stream);
    return ioData;
}
OctIOdata* ProcessGPULabviewAsync(OctGPUParams* params, OctIOdata* ioData, int processingOptions, unsigned short srcBScan[], float dstResult[])
{
    ioData->srcBscan = srcBScan;
    ioData->dstResult = dstResult;
    ProcessGPU(*params, *ioData, processingOptions);
    return ioData;
}

OctGPUParams* writeCudaBufferLabView(OctGPUParams* params, float* background, int* convInds, float* convCoef, float* conCn){
    gpuCpySyncToDevice(params->baGround_C, background, params->nbrOfSamplesAScanRaw*sizeof(float));
    gpuCpySyncToDevice(params->convInds_C, convInds, params->nbrOfSamplesAScanRaw*params->zeroPaddingFactor*params->overSamplingRatio*3*sizeof (int));
    gpuCpySyncToDevice(params->convCoeff_C, convCoef, params->nbrOfSamplesAScanRaw*params->zeroPaddingFactor*params->overSamplingRatio*3*sizeof(float));
    gpuCpySyncToHost(params->conCn_C, conCn, params->nbrOfSamplesAScanRaw*params->zeroPaddingFactor*params->overSamplingRatio*3*sizeof(float));
    return params;
}

void writeCudaConstLabview(int sizeSMStartInds, unsigned int* startInds, float* dispCoef, int windowPmin, int windowPmax, float overSamplingRatio){
    writeCudaConstSizeSMStartInds(&sizeSMStartInds);
    writeCudaConstSMStartInds(startInds, sizeSMStartInds);
    writeCudaConstDispCoeff(dispCoef, 3);
    writeCudaConstWindowPmin(&windowPmin);
    writeCudaConstWindowPmax(&windowPmax);
    writeCudaConstOverSamplingRatio(&overSamplingRatio);
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

