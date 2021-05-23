/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
#include "cuda_includes.h"

// ************* Allocation ***************************************************************************
/**
 * @brief gpuPinHostAndMallocDeviceMemory
 * @param ptr   Pointer to a memory location of the host computer that shall be pinned.
 * @param size  Size of that memory buffer.
 * @return The Device pointer to the equivalent buffer allocated on the gpu global memory.
 * @details This function pins given memory and allocates a similar buffer on the gpu's global memory.
 */
void* gpuPinHostAndMallocDeviceMemory(void* ptr, size_t size);

/**
 * @brief gpuPinHostMemory
 * @param hostPtr   Pointer to a memory location of the host computer.
 * @param size      Size of that memory buffer.
 * @details This function allocates a similar buffer on the gpu's global memory.
 */
void gpuPinHostMemory(void* hostPtr, size_t size);

/**
 * @brief gpuMallocDeviceMemory
 * @param size  Size of the buffer to allocate in bytes.
 * @return Device Pointer to the allocated memory buffer.
 */
void* gpuMallocDeviceMemory(size_t size);

/**
 * @brief gpuUnpinHostAndFreeDeviceMemory
 * @param devPtr    Device pointer to the gpu memory buffer that shall be freed.
 * @param hostPtr   Host pointer to the memory buffer on the host computer that shall be unpinned.
 */
void gpuUnpinHostAndFreeDeviceMemory(void* devPtr, void* hostPtr);

/**
 * @brief gpuUnpinHostMemory
 * @param hostPtr   Host pointer to the memory buffer that shall be unpinned.
 */
void gpuUnpinHostMemory(void* hostPtr);

/**
 * @brief gpuFreeDeviceMemory
 * @param devPtr    Device pointer to global gpu memory buffer that shall be freed.
 */
void gpuFreeDeviceMemory(void* devPtr);

// ************* Copy *********************************************************************************
/**
 * @brief gpuCpyAsyncToDevice
 * @param devPtr    Device pointer to the existing gpu global memory buffer to store the data.
 * @param hostPtr   Host pointer to the memory buffer where the data is stored.
 * @param byteSize  Number of bytes to copy.
 * @param stream    Pointer to the stream object that shall handle this copy command.
 * @param sync      Blocks execution until copy is finished if true.
 */
void gpuCpyAsyncToDevice(void* devPtr, const void* hostPtr, size_t byteSize, cudaStream_t* stream, bool sync);

/**
 * @brief gpuCpyAsyncToHost
 * @param devPtr    Device pointer to the gpu global memory buffer in which the data is stored.
 * @param hostPtr   Host pointer to the memory buffer where the data shall be stored.
 * @param byteSize  Number of bytes to copy
 * @param stream    Pointer to the stream object that shall handle this copy command.
 * @param sync      Blocks exectution until copy is finished if true.
 */
void gpuCpyAsyncToHost(const void* devPtr, void* hostPtr, size_t byteSize, cudaStream_t* stream, bool sync);

/**
 * @brief gpuCpySyncToDevice : Default stream copies data from host to gpu.
 * @param devPtr    Device pointer to the existing gpu global memory buffer to store the data.
 * @param hostPtr   Host pointer to the memory buffer where the data is stored.
 * @param byteSize  Number of bytes to copy.
 */
void gpuCpySyncToDevice(void* devPtr, const void* hostPtr, size_t byteSize);

/**
 * @brief gpuCpySyncToHost : Default stream copies data from gpu back to host.
 * @param devPtr    Device pointer to the gpu global memory buffer in which the data is stored.
 * @param hostPtr   Host pointer to the memory buffer where the data shall be stored.
 * @param byteSize  Number of bytes to copy
 */
void gpuCpySyncToHost(const void* devPtr, void* hostPtr, size_t byteSize);

// ************* Streams ******************************************************************************
/**
 * @brief gpuWaitForStream : Blocks execution until the stream has finished all enqueued work.
 * @param stream    Pointer to the stream which shall be waited for.
 */
void gpuWaitForStream(cudaStream_t* stream);

/**
 * @brief gpuGetNewStream
 * @return  Pointer to the newly created stream object. (Memory was allocated and must be freed by the user!)
 */
cudaStream_t* gpuGetNewStream(void);

/**
 * @brief gpuCleanUpStream
 * @param stream    Stream object that shall be freed.
 */
void gpuCleanUpStream(cudaStream_t* stream);

// ************* Device Info **************************************************************************
/**
 * @brief gpuGetNumberOfCUDADevices
 * @return  The number of CUDA capable devices found in the system.
 */
int gpuGetNumberOfCUDADevices(void);

/**
 * @brief gpuGetDeviceProperties    : Reads out all the specifications of the cuda device.
 * @param props     Pointer to a device protperty object.
 * @param deviceID  ID of the cuda capable device to analyze.
 */
void gpuGetDeviceProperties(cudaDeviceProp* props, int deviceID);

/**
 * @brief gpuGetGPUInfos    : Scans whole system for cuda devices and prints all the device properties to each of them.
 */
void gpuGetGPUInfos();

/**
 * @brief gpuPrintDevProp   : Prints devices properties to CLI.
 * @param devProp   Device property obeject.
 */
void gpuPrintDevProp(cudaDeviceProp devProp);
#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H

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

