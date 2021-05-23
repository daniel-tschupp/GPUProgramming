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

    string path = "bscan512x640.png";
    ImData* img = ImageHandler::loadImage(path);

    cout << "Cuda Stream example" << endl;

    cout << "Load Image data" << endl;
    shared_ptr<ImageData> imgData = make_shared<ImageData>(img->width, img->height);
    shared_ptr<ImageData> imgBG = make_shared<ImageData>(img->width, 1);
    shared_ptr<ImageData> imgOut = make_shared<ImageData>(img->width, img->height);

    // Load B-Scan from file and write it to the imput Image
    ImageHandler::extractArray(img, imgData->getImage());

    // create random Background vector
    for(uint ibg = 0; ibg < img->width; ibg++){
        float r3 = 0.1f + static_cast<float>(rand()) /( static_cast<float>(RAND_MAX/(0.3f-0.1f)));
        imgBG->getImage().at(ibg) = r3;
    }

    // Run multiple Asynchron with a Streamer Class **************************************

    // Creating the Streams and store them inside a vector
    vector<CUDA_Streamer*> streams;
    uint nStreams = 1;
    for(uint iStream = 0; iStream < nStreams; iStream++){
        streams.push_back(new CUDA_Streamer(img->height*img->width*sizeof(float)));
    }

    // Create input and output buffers for each stream, pin this buffers for fast access and
    // store the image data into the input buffers.
    vector<shared_ptr<ImageData>> imgDatas;
    vector<shared_ptr<ImageData>> imgOuts;
    for(uint iStream = 0; iStream < nStreams; iStream++){
        imgDatas.push_back(make_shared<ImageData>(img->width, img->height));
        imgOuts.push_back(make_shared<ImageData>(img->width, img->height));

        CUDA_Streamer::pinMemory(imgDatas.at(iStream)->getImagePtr(), imgDatas.at(iStream)->getByteSize() );
        CUDA_Streamer::pinMemory(imgOuts.at(iStream)->getImagePtr(), imgOuts.at(iStream)->getByteSize() );

        ImageHandler::extractArray(img, imgDatas.at(iStream)->getImage());
    }

    // Set the Background for each Stream
    for(uint iStream = 0; iStream < nStreams; iStream++){
        streams.at(iStream)->setBackground(imgBG);
    }

    // Looping through all iteratinos constantly changing the stream
    uint iStream = 0;
    Stopwatch sw2;
    sw2.Start();
    cudaProfilerStart(); // starts profiling with nvprof

    for(uint iRepetitions = 0; iRepetitions < 1000; iRepetitions++){
        streams.at(iStream)->setImage(imgDatas.at(iStream), imgOuts.at(iStream));
        streams.at(iStream)->runKernel();

        iStream++;
        if(iStream >= nStreams)
            iStream = 0;
        if(iRepetitions>nStreams-1)
            streams.at(iStream)->getResultImage();
    }

    cudaProfilerStop(); // ends profiling with nvprof
    sw2.Stop();
    cout << "4 Streamer Classes took " << sw2.GetElapsedTimeMilliseconds() << "ms to execute BG removal kernel." << endl;

    // show the nth-last of the computed output images
    ImageHandler::showArray(imgOuts[0]);

    // wait for a user key befor program ends. Otherwise it wouldn't be possible for the user to see the image.
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
