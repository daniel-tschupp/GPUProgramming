/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDAOCT_KERNELS_H
#define CUDAOCT_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_includes.h"
#include "cuda_types.h"

/**
 * @brief postFFTProcess : Function that executes the oct-processing steps after the FFT.
 * @param params Structure containing information about the OCT Processing configuration.
 * @param ioData Structure containing the pointers to all the data buffers.
 * @param scaling Value that specifies the scaling factor for the previous FFT.
 * @details Processing steps executed by this function: Remap factor multiplication conCn, Conversion from Complex to Real, FFT Shift.
 */
__global__ void postFFTProcess(OctGPUParams params, OctIOdata ioData, float scaling);

/**
 * @brief preFFTProcess : Function that exectutes the oct-processing steps needed befor the FFT can be applied.
 * @param params Structure containing information about the OCT Processing configuration.
 * @param ioData Structure containing the pointers to all the data buffers.
 * @details Processing steps executed by this function: Remove Background vector, Remove DC, Spectral Shaping, Remap, Windowing, Dispersion Compensation.
 */
__global__ void preFFTProcess(OctGPUParams params, OctIOdata ioData);

/**
 * @brief byPassFFT : This function is used when no FFT is executed to do the conversion from Complex to Real.
 * @param params Structure containing information about the OCT Processing configuration.
 * @param ioData Structure containing the pointers to all the data buffers.
 * @param scaling Value that specifies the scaling factor for the previous FFT.
 */
__global__ void byPassFFT(OctGPUParams params, OctIOdata ioData, float scaling);

#ifdef __cplusplus
}
#endif

#endif // CUDAOCT_KERNELS_H

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

