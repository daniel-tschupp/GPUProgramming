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

#include "oct_convolutionresampling.h"
#include <cmath>
#include <algorithm>
#include <numeric>

template <typename T_>
void arr_destructor(T_ buf[])
{
    delete[] buf;
}

ConvolutionResampling::ConvolutionResampling(vector<float> &remapVector, const size_t nbrOfSamplesAScan, const size_t nbrOfSamplesAScanProc, const size_t &nbrOfSamplesAScanZeroPad,
                                             const float OSR, const size_t pmin, const size_t pmax):
    m_remapVector(remapVector),                                 // remap vector
    m_Kerneltype(EKaiser),                                      // set method
    m_pmin(pmin),                                               // ROI, pmin
    m_pmax(pmax),                                               // ROI, pmax
    m_nbrOfSamplesAScan(nbrOfSamplesAScan),                     // A-Scan length
    m_nbrOfSamplesAScanProc(nbrOfSamplesAScanProc),             // A-Scan length of processed data
    m_nbrOfSamplesAScanZeroPad(nbrOfSamplesAScanZeroPad),       // A-Scan length of processed and zero padded data
    m_generateTextFile(true),                                   // generate text file
    m_OSR(OSR),                                                 // oversampling ratio OSR
    m_beta(0.0f),                                               // tuning parameter beta
    m_kernelSize(3),                                            // kernel size
    m_border(0)
{
    // set size
    m_beta = PI * sqrt(static_cast<float>(pow(m_kernelSize,2))/powf(m_OSR,2)*powf(m_OSR-1.0f/2.0f,2)- 0.8f); // optimal value for beta
    m_border = static_cast<size_t>(std::floor( static_cast<float>( m_kernelSize / 2.0f ))); // border for convolution
    createKernel();
}

ConvolutionResampling::~ConvolutionResampling()
{

}


void ConvolutionResampling::createKernel()
{
    // 1) calculate dk
    //float dk = 1.0f/m_OSR;   // -> Todo: (k_max - k_min)/ (N*a)
    float dk = static_cast<float>(m_nbrOfSamplesAScan - 1) / (m_nbrOfSamplesAScanProc - 1);

    // 1 ) create k-vector, aka sampling points with spacing dk
    std::vector<float> k(m_nbrOfSamplesAScanProc);

    //    k = linspace(1, m_nbrOfSamplesAScan, m_nbrOfSamplesAScan * m_OSR );
    int k_ind = 0;
    std::generate( k.begin(), k.end(), [&]{ return k_ind++ * dk;  });    // lambda for linspace

    // 2 ) Get neighbouring points kj's around center value k, check if abs((k - kj)/dk) <= M/2
    // 3) Calculate C_kj and remember used indices N_int_kj for convolution
    m_inds.resize(m_nbrOfSamplesAScanProc * m_kernelSize, -1);       // Note: inited with -1
    m_C.resize( m_nbrOfSamplesAScanProc * m_kernelSize , 0.0f);    // Note: inited with zeros ()

    for (size_t i = m_border; i < m_nbrOfSamplesAScanProc-m_border; ++i) // correct for zero indexing
    {
        //  Get indices of neighbouring points kj's around center value k
        std::vector<size_t> N_inds(m_kernelSize);
        findInterpolants(k[i], N_inds);

        for (size_t j = 0; j < m_kernelSize; ++j)
        {
            float kj = m_remapVector[N_inds[j]];
            float kappa = std::fabs( ( k[i] - kj) / dk );

            if( kappa  <= static_cast<float>(m_kernelSize)/2.0f )   // check condition Eq(14) -> Kappa <= M/2
            {
                size_t ind = i * m_kernelSize + j;
                m_C[ind] = getphi(kappa);     // otherwise m_C = 0!
                m_inds[ind] = N_inds[j];   // otherwise m_N_int = -1!
            }
        }
    }

    calculateCn();

}

void ConvolutionResampling::findInterpolants( float k, std::vector<size_t> &N_int)
{
    std::vector<float> diff_rk(m_remapVector.size());


    for (size_t i = 0; i < m_remapVector.size(); ++i)
    {
        diff_rk[i] = std::fabs( m_remapVector[i] - k);
    }

    // get 3 closest -> kj
    // copy m_kernelSize smallest values from diffrk to smallest
    std::vector<float> smallest(m_kernelSize);
    std::partial_sort_copy(diff_rk.begin(), diff_rk.end(), smallest.begin(), smallest.end());

    // find indice of m_kernel smallest values
    for (size_t i = 0; i < m_kernelSize; ++i)
    {
        auto it = std::find(diff_rk.begin(), diff_rk.end(), smallest.at(i));
        N_int[i] = std::distance(diff_rk.begin(), it) ;
    }
}

void ConvolutionResampling::calculateCn()
{
    m_Cn.resize( m_nbrOfSamplesAScanZeroPad, 0.0f);
    std::vector<int> n(m_nbrOfSamplesAScanZeroPad);

    // Generate Vecotr from -Na/2 to Na/2 without 0 (zero)

    std::iota( n.begin(), n.end(), - static_cast<int>(m_nbrOfSamplesAScanZeroPad)/2);

    for(size_t i=0; i<m_nbrOfSamplesAScanZeroPad; ++i )
    {
        std::complex<float> z =std::sqrt( static_cast<std::complex<float>>( powf( n.at(i) * PI * m_kernelSize / m_nbrOfSamplesAScanZeroPad, 2) - powf(m_beta,2) ) );
        m_Cn[i] = (z/std::sin(z)).real();
    }

    // do "ffsthift" to align the apodization function to the fftshifted data!
    std::rotate(m_Cn.begin(), m_Cn.begin() + (m_nbrOfSamplesAScanZeroPad/2), m_Cn.end());
}

float ConvolutionResampling::getphi(float x)
{
    float xAbs = fabs(x);
    float phi;

    switch(m_Kerneltype)
    {

    case ELinear:
        if (xAbs < 1)
        {
            phi = 1.0f - fabs(x);
        }
        else
        {
            phi = 0.0f;
        }
        break;

    case ECubic:          // cubic spline
        if (xAbs < 1.0f)
        {
            phi = 2.0f/3.0f - xAbs*xAbs + 0.5f*powf(xAbs,3);
        }
        else if (xAbs < 2.0f)
        {
            phi = (1.0f/6.0f)*powf(2.0f-xAbs,3);

        }
        else
        {
            phi = 0.0f;
        }
        break;

    case EKeysCubic:
        if (xAbs < 1.0f)
        {
            phi = 1.0f  - 2.5f*powf(xAbs,2) + 1.5f*powf(xAbs,3);

        }
        else if (xAbs < 2)
        {
            phi = -0.5f*(-4.0f + 8.0f*xAbs -5.0f*powf(xAbs,2) + powf(xAbs,3));
        }
        else
        {
            phi = 0.0f;
        }
        break;

    case ECatmull:
        if(xAbs < 1.0f)
        {
            phi = 1.0f - (5.0f/2.0f)*powf(xAbs,2) + (3.0f/2.0f)*powf(xAbs,3);
        }
        else if (xAbs < 2)
        {
            phi = 2.0f - 4.0f*xAbs + (5.0f/2.0f)*powf(xAbs,2) -(1.0f/2.0f)*powf(xAbs,3);
        }
        else
        {
            phi = 0.0f;
        }
        break;

    case EGauss:
        phi = exp(-(m_beta*powf(x,2)));
        break;

    case EKaiser:
    {
        phi = alglib::besseli0(m_beta * sqrtf( 1.0f - powf( 2.0f*xAbs / m_kernelSize, 2) ) ) / m_kernelSize;
    }
        break;

    case ESinc:
        phi = sin(PI*xAbs)/(PI*xAbs);
        break;

    default:
        phi = 0.0f;
        break;
    }
    return phi;
}
