#include "imagedata.h"

ImageData::ImageData(size_t width, size_t height):
    m_ImWidth(width),
    m_ImHeight(height),
    m_ImSize(width*height*sizeof(float))
{
    for(unsigned int i = 0; i<width*height;i++)
        m_ImData.push_back(0);
}

float* ImageData::getImagePtr(void){
    return m_ImData.data();
}
vector<float>& ImageData::getImage(void){
    return m_ImData;
}
size_t ImageData::getByteSize(void){
    return m_ImSize;
}
size_t ImageData::getWidth(void){
    return m_ImWidth;
}
size_t ImageData::getHeight(void){
    return m_ImHeight;
}
