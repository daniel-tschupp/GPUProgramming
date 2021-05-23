#include <iostream>
#include <string>
#include "imagehandler.h"

#include <cuda.h>
#include "device_launch_parameters.h"
#include "kernels.cuh"

#include "cuda_profiler_api.h"

using namespace std;

void removeBackground_original(float* inBScan_H, float* outBScan_H, float* bg_H, unsigned int width, unsigned int height);

int main()
{
    cudaFree(nullptr);
    CUDA_DeviceHelper helper(0, true);
    //Loading B-Scan
    string path = "bscan512x640.png";
    ImData* img = ImageHandler::loadImage(path);
    float* inData = ImageHandler::extractArray(img);
    float* outData = new float[img->width*img->height];

    // Show original B-Scan
    ImageHandler::showArray(inData, img->width, img->height);

    float* bg = new float[img->width];
    for(uint ibg = 0; ibg < img->width; ibg++){
        float r3 = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.2f-0.1f)));
        bg[ibg] = r3;
    }

    // Measure differen background subtraction kernels
    cout << "Measure differen background subtraction kernels:" << endl << endl;

    cudaProfilerStart();
    removeBackground_original(inData, outData, bg, img->width, img->height);
    removeBackground_sharedMemory(inData, outData, bg, img->width, img->height);
    removeBackground_BGsharedMemory(inData, outData, bg, img->width, img->height);
    removeBackground_constMemory(inData, outData, bg, img->width, img->height);
    removeBackground_cachingBG(inData, outData, bg, img->width, img->height);
    removeBackground_dynamicParallelism(inData, outData, bg, img->width, img->height);
    removeRowBackground_constMemory(inData, outData, bg, img->width, img->height);
    removeBackground_optimBurst(inData, outData, bg, img->width, img->height);
    cudaProfilerStop();

    ImageHandler::showArray(outData, img->width, img->height);

    waitKey(0);

    // Clean up image
    delete img;
    delete[] inData;
    delete[] outData;
    delete[] bg;
    return 0;
}


