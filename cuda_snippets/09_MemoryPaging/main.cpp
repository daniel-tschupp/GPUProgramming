#include <iostream>
#include <string>
#include "imagehandler.h"

#include <cuda.h>
#include "device_launch_parameters.h"
#include "kernels.cuh"

using namespace std;

void removeBackground_original(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height);

int main()
{
    // Allocating normal Memory and changing it to paging:
    cudaFree(0);
    //Loading B-Scan
    string path = "bscan512x640.png";
    ImData* img = ImageHandler::loadImage(path);
    float* inData = ImageHandler::extractArray(img);
    float* outData = new float[img->width*img->height];

    // Show original B-Scan
    //ImageHandler::showArray(inData, img->width, img->height);

    float* bg = new float[img->width];
    for(uint ibg = 0; ibg < img->width; ibg++){
        float r3 = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f-0.1f)));
        bg[ibg] = r3;
    }

    // Measure differen background subtraction kernels
    cout << "Measure differen background subtraction kernels:" << endl << endl;

    cout << "Measure normal memory transfer:" << endl;
    removeBackground_cachingBG(inData, outData, bg, img->width, img->height);

    ImageHandler::showArray(outData, img->width, img->height);
    waitKey(0);
    //*******************************************************************************************************************
    // Change normal memory to Paged memory

    cudaHostRegister(inData, img->height*img->width*sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(bg, img->width*sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(outData, img->height*img->width*sizeof(float), cudaHostRegisterDefault);

    //*******************************************************************************************************************
    cout << "Measure paged memory transfer: (default paging)" << endl;
    removeBackground_cachingBG(inData, outData, bg, img->width, img->height);

    // Clean up image
    cudaFreeHost(inData);
    cudaFreeHost(outData);
    cudaFreeHost(bg);

    ImageHandler::showArray(outData, img->width, img->height);
    delete inData;
    delete outData;
    delete bg;
    waitKey(0);

    //*******************************************************************************************************************
    // Pud data in special Write-Combining Memory Paging for faster transfer (BUT now reading from this memory is slow for the Host)
    cout << "Measure paged memory transfer: (write Combining):" << endl;
    cudaHostAlloc((void**)&inData, img->height*img->width*sizeof(float), cudaHostAllocWriteCombined);
    cudaHostAlloc((void**)&bg, img->width*sizeof(float), cudaHostAllocWriteCombined);
    cudaHostAlloc((void**)&outData, img->height*img->width*sizeof(float), cudaHostAllocDefault);

    ImageHandler::extractArray(img, inData);
    for(uint ibg = 0; ibg < img->width; ibg++){
        float r3 = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f-0.1f)));
        bg[ibg] = r3;
    }
    removeBackground_cachingBG(inData, outData, bg, img->width, img->height);

    ImageHandler::showArray(outData, img->width, img->height);

    cudaFreeHost(inData);
    cudaFreeHost(outData);
    cudaFreeHost(bg);
    waitKey(0);

    //*******************************************************************************************************************
    // Use Host Pointer For Registered Memory with architectures > 3.5
    CUDA_DeviceHelper devHelper(0, false);
    cout << "Can Use Host Pointer For Registered Memory: " << devHelper.getDevProps().canUseHostPointerForRegisteredMem << endl;
    if(devHelper.getDevProps().canUseHostPointerForRegisteredMem != 0){
        cudaSetDeviceFlags(cudaDeviceMapHost);

        cudaHostAlloc((void**)&inData, img->height*img->width*sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&bg, img->width*sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&outData, img->height*img->width*sizeof(float), cudaHostAllocMapped);

        ImageHandler::extractArray(img, inData);
        for(uint ibg = 0; ibg < img->width; ibg++){
            float r3 = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f-0.1f)));
            bg[ibg] = r3;
        }
        removeBackground_cachingBG_Mapped(inData, outData, bg, img->width, img->height);
        ImageHandler::showArray(outData, img->width, img->height);
        cudaFreeHost(inData);
        cudaFreeHost(outData);
        cudaFreeHost(bg);
    }

    //*******************************************************************************************************************
    // Directely access host memory with architectures >= 7.0
    if(ARCH >= 70){
        cout << "Compute Capability >= 7.0 can directely access host memory" << endl;
        inData = static_cast<float*>(malloc(img->height*img->width*sizeof(float)));
        bg = static_cast<float*>(malloc(img->width*sizeof(float)));
        outData = static_cast<float*>(malloc(img->height*img->width*sizeof(float)));

        ImageHandler::extractArray(img, inData);
        for(uint ibg = 0; ibg < img->width; ibg++){
            float r3 = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f-0.1f)));
            bg[ibg] = r3;
        }

        removeBackground_cachingBG_Mapped(inData, outData, bg, img->width, img->height);

        ImageHandler::showArray(outData, img->width, img->height);
        free(inData);
        free(outData);
        free(bg);
    }


    delete img;
    waitKey(0);
    return 0;
}


