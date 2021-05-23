#ifndef CUDA_STREAMER_H
#define CUDA_STREAMER_H

#include <memory>
#include "Kernels/kernel_utils.h"
#include "Kernels/kernels.cuh"
#include "imagedata.h"

using namespace std;

class CUDA_Streamer
{
    cudaStream_t* mpStream;

    shared_ptr<ImageData> mpInputImage;
    shared_ptr<ImageData> mpBackgroundImage;
    shared_ptr<ImageData> mpOutputImage;

    float* dInputImage;
    float* dBackgroundImage;
    float* dOutputImage;

    size_t mDataByteSize;
public:
    CUDA_Streamer(size_t dataByteSize);
    ~CUDA_Streamer();
    static void pinMemory(float* hostPtr, size_t size);
    static void unpinMemory(float* hostPtr);
    float* allocateDeviceMemory(size_t size);
    void freeDeviceMemory(float* devPtr);
    void setImage(shared_ptr<ImageData> pInputImage, shared_ptr<ImageData> pOutputImage);
    void setBackground(shared_ptr<ImageData> pImage);
    shared_ptr<ImageData> getResultImage(void);
    void runKernel(void);

};

#endif // CUDA_STREAMER_H
