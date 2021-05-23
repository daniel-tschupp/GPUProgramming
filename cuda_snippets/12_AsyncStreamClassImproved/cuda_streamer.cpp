#include "cuda_streamer.h"

using namespace std;

CUDA_Streamer::CUDA_Streamer(size_t dataByteSize):
    mpInputImage(make_shared<ImageData>(1,1)),
    mpBackgroundImage(make_shared<ImageData>(1,1)),
    mpOutputImage(make_shared<ImageData>(1,1)),
    mDataByteSize(dataByteSize)
{
    mpStream = getNewStream();
    dInputImage = this->allocateDeviceMemory(mDataByteSize);
    dOutputImage = this->allocateDeviceMemory(mDataByteSize);

}
CUDA_Streamer::~CUDA_Streamer(){
    unpinHostAndFreeDeviceMemory(dBackgroundImage, mpBackgroundImage->getImagePtr() );
    unpinHostAndFreeDeviceMemory(dInputImage, mpInputImage->getImagePtr() );
    unpinHostAndFreeDeviceMemory(dOutputImage, mpOutputImage->getImagePtr() );

    free(mpStream);
}

void CUDA_Streamer::pinMemory(float* hostPtr, size_t size){
    pinHostMemory(hostPtr, size);
}
void CUDA_Streamer::unpinMemory(float* hostPtr){
    unpinHostMemory(hostPtr);
}

float* CUDA_Streamer::allocateDeviceMemory(size_t size){
    return mallocDeviceMemory(size);
}
void CUDA_Streamer::freeDeviceMemory(float* devPtr){
    freeDeviceMemory(devPtr);
}


void CUDA_Streamer::setImage(shared_ptr<ImageData> pInputImage, shared_ptr<ImageData> pOutputImage){
    mpInputImage = pInputImage;
    mpOutputImage = pOutputImage;/*make_shared<ImageData>(pImage->getWidth(), pImage->getHeight());
    dInputImage = pinHostAndMallocDeviceMemory( mpInputImage->getImagePtr(),
                                                mpInputImage->getByteSize() );
    dOutputImage = pinHostAndMallocDeviceMemory(    mpOutputImage->getImagePtr(),
                                                    mpOutputImage->getByteSize() );*/
}
void CUDA_Streamer::setBackground(shared_ptr<ImageData> pImage){
    mpBackgroundImage = pImage;
    dBackgroundImage = pinHostAndMallocDeviceMemory(    mpBackgroundImage->getImagePtr(),
                                                        mpBackgroundImage->getByteSize() );
    cpyAsyncToDevice(dBackgroundImage, mpBackgroundImage->getImagePtr(), mpBackgroundImage->getByteSize(), mpStream);
}
shared_ptr<ImageData> CUDA_Streamer::getResultImage(void){
    waitForStream(mpStream);
    //unpinHostAndFreeDeviceMemory(dInputImage, mpInputImage->getImagePtr() );
    //unpinHostAndFreeDeviceMemory(dOutputImage, mpOutputImage->getImagePtr() );
    return mpOutputImage;
}

void CUDA_Streamer::runKernel(void){

    cpyAsyncToDevice(dInputImage, mpInputImage->getImagePtr(), mpInputImage->getByteSize(), mpStream);

    // Kernel missing for the moment.
    BackgroundKernelParams params;
    params.blockWidth = 32;
    params.bScanHeight = mpInputImage->getHeight();
    params.bScanWidth = mpInputImage->getWidth();
    params.dBScanIn = dInputImage;
    params.dBackground = dBackgroundImage;
    params.dBScanOut = dOutputImage;

    executeBGRemoveKernel(mpStream, params);

    cpyAsyncToHost(dOutputImage, mpOutputImage->getImagePtr(), mpOutputImage->getByteSize(), mpStream);
}

