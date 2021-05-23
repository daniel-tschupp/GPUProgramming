#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include "memory.h"

#include "utils/Stopwatch.h"
#include "utils/imagehandler.h"
#include "cuda_streamer.h"
#include "imagedata.h"

#include "cuda_profiler_api.h"

using namespace std;

int main()
{
    // Allocating normal Memory and changing it to paging:
    cudaFree(nullptr);

    cout << "Cuda Stream example" << endl;
    string path = "bscan512x640.png";
    ImData* img = ImageHandler::loadImage(path);

    // Create input / output / background images
    shared_ptr<ImageData> imgData = make_shared<ImageData>(img->width, img->height);
    shared_ptr<ImageData> imgBG = make_shared<ImageData>(img->width, 1);
    shared_ptr<ImageData> imgOut = make_shared<ImageData>(img->width, img->height);

    // insert the image data from the file into the input image and create a random background vector
    cout << "Load Image data" << endl;
    ImageHandler::extractArray(img, imgData->getImage());
    for(uint ibg = 0; ibg < img->width; ibg++){
        float r3 = 0.1f + static_cast<float>(rand()) /( static_cast<float>(RAND_MAX/(0.3f-0.1f)));
        imgBG->getImage().at(ibg) = r3;
    }

    // Run kernel asynchronous with a Stream **************************************

    cudaStream_t* stream = getNewStream();

    float* imgIn_D = pinHostAndMallocDeviceMemory(imgData->getImagePtr(), imgData->getByteSize());
    float* imgBG_D = pinHostAndMallocDeviceMemory(imgBG->getImagePtr(), imgBG->getByteSize());

    cpyAsyncToDevice(imgIn_D, imgData->getImagePtr(), imgData->getByteSize(), stream);
    cpyAsyncToDevice(imgBG_D, imgBG->getImagePtr(), imgBG->getByteSize(), stream);

    cpyAsyncToHost(imgIn_D, imgOut->getImagePtr(), imgData->getByteSize(), stream);
    waitForStream(stream);


    // Test Streamer Class *********************************************************

    CUDA_Streamer streamer;
    Stopwatch sw1;

    streamer.setBackground(imgBG);

    sw1.Start();
    for(int i = 0; i < 1000; i++){
        streamer.setImage(imgData);
        streamer.runKernel();
        imgOut = streamer.getResultImage();
    }
    sw1.Stop();

    ImageHandler::showArray(imgOut);
    cout << "Streamer Class took " << sw1.GetElapsedTimeMilliseconds() << "ms to execute BG removal kernel." << endl;


    // Test multiple Streamers *********************************************************

    // Create multiple streams
    vector<CUDA_Streamer*> streams;
    uint nStreams = 10;
    for(uint iStream = 0; iStream < nStreams; iStream++){
        streams.push_back(new CUDA_Streamer());
    }

    // Create in / out buffers for every stream
    vector<shared_ptr<ImageData>> imgDatas;
    vector<shared_ptr<ImageData>> imgOuts;
    for(uint iStream = 0; iStream < nStreams; iStream++){
        imgDatas.push_back(make_shared<ImageData>(img->width, img->height));
        imgOuts.push_back(make_shared<ImageData>(img->width, img->height));

        ImageHandler::extractArray(img, imgDatas.at(iStream)->getImage());
    }

    // Set Background for each stream
    for(uint iStream = 0; iStream < nStreams; iStream++){
        streams.at(iStream)->setBackground(imgBG);
    }

    // Cycle through all the repetitions
    uint iStream = 0;
    Stopwatch sw2;
    sw2.Start();
    cudaProfilerStart();
    for(uint iRepetitions = 0; iRepetitions < 1000; iRepetitions++){
        streams.at(iStream)->setImage(imgDatas.at(iStream));
        streams.at(iStream)->runKernel();

        iStream++;
        if(iStream >= nStreams)
            iStream = 0;
        if(iRepetitions>3)
            streams.at(iStream)->getResultImage();
    }
    cudaProfilerStop();
    sw2.Stop();

    cout << "4 Streamer Classes took " << sw2.GetElapsedTimeMilliseconds() << "ms to execute BG removal kernel." << endl;

    // Wait for user entry to end the program.
    waitKey(0);
    return 0;
}

void readMatrix(float* const pM, const int size_M, const string filename){
    ifstream inputData(filename);
    for(int i = 0; i<size_M; i++){
        string tmp;
        inputData >> tmp;
        pM[i] = std::stof(tmp);
    }
}

void createMatrix(float* const pM, const int size_M, const string filename){
    srand((unsigned)time(0));
    for(int i= 0; i < size_M; i++)
        pM[i] = static_cast <float> (rand())/static_cast <float>(20000);

    ofstream arrayData(filename); // File Creation(on C drive)

    for(int k=0;k<size_M;k++)
    {
        arrayData<<pM[k]<<endl;
    }
    cout << "Created File: " << filename << " with " << size_M << " Elements." << endl;
}

void showMatrix(const float* const pM, const int columns, const int rows){
    cout << "Matrix:" << endl;

    for(int i = 0; i < rows; i++){
        for(int k = 0; k < columns; k++)
            cout << pM[columns*i + k] << "\t";
        cout << endl;
    }
}
