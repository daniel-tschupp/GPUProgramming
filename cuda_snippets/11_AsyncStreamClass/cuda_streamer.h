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

public:
    CUDA_Streamer();
    ~CUDA_Streamer();
    void setImage(shared_ptr<ImageData> pImage);
    void setBackground(shared_ptr<ImageData> pImage);
    shared_ptr<ImageData> getResultImage(void);
    void runKernel(void);

};

#endif // CUDA_STREAMER_H
