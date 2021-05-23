/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDA_ERRORCHECKING_H
#define CUDA_ERRORCHECKING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "cuda_includes.h"

// Activate devines for Debugging
//#define DEBUG_STARTINDS
//#define DEBUG_KERNEL_SIZE
//#define DEBUG_GLOB_ALLOC_MEM
//#define DEBUG_OPTIONS
//#define DEBUG_GPU_SPECS
//#define DEBUG_KERNEL_BOUNDARIES
//#define MODULE_CUDA_DEBUG
//#define CUDA_DEBUG_MSG

/** @brief This Makro gives a proper error in the CLI when a cuda API function produces an error. */
#define CudaSafeAPICall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
/** @brief This Makro gives a proper error in the CLI when a cuda cuFFT function produces an error. */
#define CudaSafeFFTCall( res ) __cudaSafeFFTCall( res, __FILE__, __LINE__ )
/** @brief This Makro gives a proper error in the CLI for an index out of bound error inside a kernel. */
#define CudaCheckKernelError()    __cudaCheckError( __FILE__, __LINE__ )

/**
 * @brief __cudaSafeCall: This function gives a proper error in the CLI when a cuda API function produces an error. It is only active if the MODULE_CUDA_DEBUG flag is set.
 * @param res Result of the API function.
 * @param file File in which the function is called.
 * @param line Line at which the function is called.
 */
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef MODULE_CUDA_DEBUG
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CudaSafeAPICall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

/**
 * @brief __cudaSafeFFTCall: This function gives a proper error in the CLI when a cuFFT function produes an error. It is only active if the MODULE_CUDA_DEBUG flag is set.
 * @param res Result of the cuFFT function.
 * @param file File in which the function is called.
 * @param line Line at which the function is called.
 */
inline void __cudaSafeFFTCall( cufftResult res, const char *file, const int line )
{
#ifdef MODULE_CUDA_DEBUG
    if ( CUFFT_SUCCESS != res )
    {
        fprintf( stderr, "CudaSafeFFTCall() failed at %s:%i\n",
                 file, line);
        exit( -1 );
    }
#endif

    return;
}
/**
 * @brief __cudaCheckError: This function gives a proper error in the CLI for a Index out of bound error inside a kernel. It is only active if the MODULE_CUDA_DEBUG flag is set.
 * @param file File in which the function is called.
 * @param line Line at which the function is called.
 */
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef MODULE_CUDA_DEBUG
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CudaCheckKernelError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}



#ifdef __cplusplus
}
#endif

#endif // CUDA_ERRORCHECKING_H

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

