#include "imagehandler.h"
#include <string>


ImageHandler::ImageHandler()
{

}
ImageHandler::~ImageHandler(){

}
ImData* ImageHandler::loadImage(string path){
    Mat image;
    Mat grey;
   image = imread(path, 1 );

    if ( !image.data )
    {
        printf("No image data \n");
    }

    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    cv::cvtColor(image, grey, cv::COLOR_RGB2GRAY);

    ImData* imdata = new ImData();
    imdata->mat = new Mat(grey);
    imdata->width = image.size[1];
    imdata->height = image.size[0];
    cout << "Image w x h: " << imdata->width << " x " << imdata->height << endl;

    return imdata;
}
float* ImageHandler::extractArray(ImData* imdata){
    unsigned int w = imdata->width;
    unsigned int h = imdata->height;
    Mat* mat = imdata->mat;

    float* dataPtr = new float[w*h];
    for(unsigned int iRow=0;  iRow < h; iRow++){
        for(unsigned int iCol=0; iCol < w; iCol++){
            uchar tmp = mat->data[iRow*w+iCol];
            dataPtr[iRow*w+iCol] = (tmp/255.0f);
        }
    }
    return dataPtr;
}

void ImageHandler::extractArray(ImData* imdata, float* data){
    unsigned int w = imdata->width;
    unsigned int h = imdata->height;
    Mat* mat = imdata->mat;

    for(unsigned int iRow=0;  iRow < h; iRow++){
        for(unsigned int iCol=0; iCol < w; iCol++){
            uchar tmp = mat->data[iRow*w+iCol];
            data[iRow*w+iCol] = (tmp/255.0f);
        }
    }
}

void ImageHandler::showArray(float* array, unsigned int width, unsigned int height){

    cv::Mat A(height, width, CV_32F, array);

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", A);
}
