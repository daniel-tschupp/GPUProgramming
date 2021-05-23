#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <memory>
#include "../imagedata.h"

using namespace std;
using namespace cv;

class ImData{
public:
    unsigned int width;
    unsigned int height;
    Mat* mat;
    ~ImData(){delete mat;}
};


class ImageHandler
{
    ImageHandler();
    ~ImageHandler();
public:
    static ImData* loadImage(string path);
    static float* extractArray(ImData* mat);
    static void extractArray(ImData* imdata, float* data);
    static void showArray(float* array, unsigned int width, unsigned int height);
    static void showArray(shared_ptr<ImageData> Image);
    static void extractArray(ImData* imdata, vector<float>& data);
};

#endif // IMAGEHANDLER_H
