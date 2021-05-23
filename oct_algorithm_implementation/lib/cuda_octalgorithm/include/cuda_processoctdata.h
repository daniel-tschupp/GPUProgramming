/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDA_PROCESSOCTDATA_H
#define CUDA_PROCESSOCTDATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_utils.h"
#include "cuda_types.h"
#include "cuda_errorchecking.h"

/**
 * @brief ProcessGPU : Function that executes the OCT-Processing.
 * @param params Structure containing information about the OCT Processing configuration.
 * @param ioData Structure containing the pointers to all the data buffers.
 */
void ProcessGPU(OctGPUParams params, OctIOdata ioData, int processingOptions);

/**
 * @brief InitConfigCudaBuffers : Function that initializes all the buffers needed by all streams for the oct-processing on the global gpu memory.
 * @param params Structure containing information about the OCT Processing configuration.
 * @details Initialized buffers: Remap buffers, background buffer.
 */
void InitConfigCudaBuffers(OctGPUParams* params);

/**
 * @brief InitDataCudaBuffers : Function that initializes all the buffers needed by one stream for the oct-processing of this stream on the global gpu memory.
 * @param ioData Structure containing the pointers to all the data buffers.
 * @param params Structure containing information about the OCT Processing configuration.
 * @details Initialized buffers: input data buffer, output data buffer and 2 intermediate data buffers.
 */
void InitDataCudaBuffers(OctIOdata* ioData, const OctGPUParams params);

/**
 * @brief FreeCudaConfigBuffers : Clean-up of all the memory buffers used by all streams on the global gpu memory.
 * @param params Structure containing information about the OCT Processing configuration.
 */
void FreeCudaConfigBuffers(OctGPUParams* params);

/**
 * @brief FreeCudaDataBuffers : Clean-up of all the memory buffers used by one stream on th global gpu memory.
 * @param ioData Structure containing the pointers to all the data buffers.
 */
void FreeCudaDataBuffers(OctIOdata* ioData);

/**
 * @brief writeCudaBuffer : Writes data to a already allocated global gpu memory buffer.
 * @param gpuBuffer     : Pointer to the global gpu memory buffer.
 * @param hostBuffer    : Pointer to the host memory buffer.
 * @param nbrOfBytes    : Number of bytes to copy.
 */
void writeCudaBuffer(void* gpuBuffer, void* hostBuffer, unsigned int nbrOfBytes);

/**
 * @brief writeCudaConstProcessingOptions : Function to write to the gpu constant memory of the constProcessingOptions variable.
 * This Variable desiced which processing steps will be executed during a A-Scan processing.
 * @param proc_options
 */
void writeCudaConstProcessingOptions(int* proc_options);

/**
 * @brief writeCudaConstSizeSMStartInds : Function to write to the gpu constant memory of the sizeSMStartInds variable.
 * @param sizeSMStartInds
 */
void writeCudaConstSizeSMStartInds(int* sizeSMStartInds);

/**
 * @brief writeCudaConstSMStartInds : Function to write to the gpu constant memory of the smStartInds array.
 * @param startInds
 * @param elements  : Max 256 elements (for the moment)
 */
void writeCudaConstSMStartInds(unsigned int* startInds, unsigned int elements);

/**
 * @brief writeCudaConstDispCoeff   : Function to write to the gpu constant memory of the writeCudaConstDispCoeff array.
 * @param dispCoef
 * @param elements  : Number of elements must be 3 (for the moment)
 */
void writeCudaConstDispCoeff(float* dispCoef, unsigned int elements);

/**
 * @brief writeCudaConstWindowPmin  : Function to write to the gpu constant memory of the windowPmin variable.
 * @param windowPmin
 */
void writeCudaConstWindowPmin(int* windowPmin);

/**
 * @brief writeCudaConstWindowPmax  : Function to write to the gpu constant memory of the windowPmax variable.
 * @param windowPmax
 */
void writeCudaConstWindowPmax(int* windowPmax);

/**
 * @brief writeCudaConstOverSamplingRatio   : Function to write to the gpu constant memory of the alpha variable.
 * @param overSamplingRatio
 */
void writeCudaConstOverSamplingRatio(float* overSamplingRatio);

/* Labview interface Section ****************************************************************************************/
// IMPORTANT: When in Labview importet change all special data types inside the cuda_types.h to standart types!
// IMPORTANT 2: Activate the checkbox of the dstResult and srcBScan convert pointer to value inside the wrapper for the ioData structure.
OctIOdata* ProcessGPULabview(OctGPUParams* params, OctIOdata* ioData, int processingOptions, unsigned short srcBScan[], float dstResult[]);
OctIOdata* ProcessGPULabviewAsync(OctGPUParams* params, OctIOdata* ioData, int processingOptions, unsigned short srcBScan[], float dstResult[]);
OctGPUParams* writeCudaBufferLabView(OctGPUParams* params, float background[], int convInds[], float convCoef[], float conCn[]);
void writeCudaConstLabview(int sizeSMStartInds, unsigned int startInds[], float dispCoef[], int windowPmin, int windowPmax, float overSamplingRatio);
OctIOdata* createNewStreamPtrLabView(OctIOdata* ioData);
OctIOdata*  cleanUpStreamLabView(OctIOdata* ioData);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PROCESSOCTDATA_H

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

