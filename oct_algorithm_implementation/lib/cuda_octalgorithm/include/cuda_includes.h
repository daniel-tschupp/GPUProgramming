/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef CUDA_INCLUDES_H
#define CUDA_INCLUDES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <cufft.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

/** @brief Makro for cudaMemcpyDeviceToHost. */
#define D2H cudaMemcpyDeviceToHost
/** @brief Makro for cudaMemcpyHostToDevice. */
#define H2D cudaMemcpyHostToDevice
/** @brief Maktro to saturate a value to a minimal value. */
#define satMin(x, min)   x<min?min:x;

#ifdef __cplusplus
}
#endif

#endif // CUDA_INCLUDES_H

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

