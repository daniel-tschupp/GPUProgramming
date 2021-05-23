#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <vector>

using namespace std;

class ImageData{
    vector<float> m_ImData;
    size_t m_ImWidth;
    size_t m_ImHeight;
    size_t m_ImSize;
public:
    ImageData(size_t width, size_t height);
    float* getImagePtr(void);
    vector<float>& getImage(void);
    size_t getByteSize(void);
    size_t getWidth(void);
    size_t getHeight(void);
};

#endif // IMAGEDATA_H
