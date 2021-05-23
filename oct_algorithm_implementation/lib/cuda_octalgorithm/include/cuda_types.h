/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_includes.h"

#define IS_DISPCOMP_ACTIVE(POption)             POption & 1
#define IS_WINDOWING_ACTIVE(POption)            POption & 2
#define IS_REMAP_ACTIVE(POption)                POption & 4
#define IS_DCREMOVE_ACTIVE(POption)             POption & 8
#define IS_BGREMOVE_ACTIVE(POption)             POption & 16
#define IS_FFT_ACTIVE(POption)                  POption & 32
#define IS_SPECTRALSHAPPING_ACTIVE(POption)     POption & 64
#define IS_CHANGE_ENDIAN(POption)               POption & 128

/** @brief ProcStates
  * @details ProcStates is a container all the constant memories.
  */
/*typedef struct _ProcStates{
    int processingOptions;
    float dispCoeff_a2;
    float dispCoeff_a3;
    float dispCoeff_a4;
    float OSR_alpha;
    size_t pmin;
    size_t pmax;
    size_t nbrSamplesCropped;
    size_t zeroPaddingFactor;
}ProcStates;*/

/** @brief OctGPUParams
  * @details OctGPUParams is a container data type to store all the necessary information to execute the oct-processing.
  */
typedef struct _octGPUParams{
    float* conCn_C;
    float* baGround_C;
    float* convCoeff_C;
    int* convInds_C;
    unsigned int nbrOfSamplesAScanRaw;
    float overSamplingRatio;
    float zeroPaddingFactor;
    unsigned int nbrOfAScans;
    unsigned int kernelSize;
    unsigned int dcKernelSize;
    unsigned int tileWidth;
    unsigned int padLeft;
    unsigned int padRight;
    float bgMean;
}OctGPUParams;

/** @brief OctIOdata
  * @details OctIOData is a container data type to store all the data buffer pointer used by one stream.
  */
typedef struct _OctIOdata{
    cudaStream_t* stream;
    float* bScanRes_C;
    float* dstResult;
    unsigned short* srcBscan;
    unsigned short* bScanRaw_C;
    cufftComplex* bScanCom_C;
    cufftComplex* bScanPreFFTResult_C;
    cufftComplex* bScanPreFFTResultRemap_C;
    cufftHandle fftHandleResult_C;
    cufftHandle fftHandleResultRemap_C;
    _OctIOdata(): stream(nullptr), bScanRes_C(nullptr), dstResult(nullptr), srcBscan(nullptr), bScanRaw_C(nullptr), bScanPreFFTResultRemap_C(nullptr), bScanCom_C(nullptr), bScanPreFFTResult_C(nullptr), fftHandleResult_C(0),fftHandleResultRemap_C(0){}
}OctIOdata;


#ifdef __cplusplus
}
#endif

#endif // CUDA_TYPES_H

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

