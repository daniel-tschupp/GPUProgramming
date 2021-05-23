/****************************************************************************
** Copyright (c) 2016 Advanced Oesteotomy Tools AG &
**                    HuCE-optoLab - Bern University of Applied Sciences
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
**
** Contact: Adrian Schneider ( a.schneider@aot-swiss.ch )
**          Patrik Arnold ( patrik.arnold@bfh.ch )
*****************************************************************************/
#ifndef CONVOLUTIONRESAMPLING_H
#define CONVOLUTIONRESAMPLING_H

#include "interpolation.h"
#include <complex>
#include <memory>
#include <vector>

using namespace std;

class ConvolutionResampling
{


public:

    enum EKerneltype
    {
        ELinear=0,
        ECubic=1,
        EKeysCubic=2,
        ECatmull=3,
        EGauss=4,
        EKaiser=5,
        ESinc=6
    };


    ConvolutionResampling(vector<float> &remapVector , const size_t nbrOfSamplesAScan, const size_t nbrOfSamplesAScanProc, const size_t &nbrOfSamplesAScanZeroPad, const float OSR, const size_t pmin, const size_t pmax);
    ~ConvolutionResampling();


    std::vector<int> getInds() const {return m_inds; }
    std::vector<float> getC() const {return m_C; }
    std::vector<float> getCn() const {return m_Cn; }
    size_t getKernelSize() const { return m_kernelSize; }
    void setKernelSize( size_t kernelSize ){ m_kernelSize = kernelSize; }
    float getOSR() const { return m_OSR; }
    void setOSR( int OSR ){ m_OSR = OSR; }

    size_t getnbrOfSamplesAScanProc() const {return m_nbrOfSamplesAScanProc;}
    size_t getBorder() const{return m_border;}

private:
    void createKernel();
    void calculateCn();
    void findInterpolants(float k, std::vector<size_t> &N_int);
    float getphi(float x);


private:
    const float PI = 3.141592653589793f;
    vector<float> m_remapVector;
    EKerneltype m_Kerneltype;
    size_t m_pmin;
    size_t m_pmax;
    size_t m_nbrOfSamplesAScan;
    size_t m_nbrOfSamplesAScanProc;
    size_t m_nbrOfSamplesAScanZeroPad;
    bool m_generateTextFile;
    float m_OSR;
    float m_beta;
    size_t m_kernelSize;
    size_t m_border;


    vector<float> m_FD;
    vector<float> m_C;
    vector<float> m_Cn;
    vector<int> m_inds;


};

#endif // CONVOLUTIONRESAMPLING_H
