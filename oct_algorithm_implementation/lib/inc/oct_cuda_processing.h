/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef OCT_CUDA_PROCESSING_H
#define OCT_CUDA_PROCESSING_H

#include <vector>
#include <mutex>
#include "i_oct_processing.h"

#include "cuda_includes.h"
#include "cuda_types.h"

class Oct_Settings;
class ConvolutionResampling;

/**
 * \class Oct_Cuda_Processing
 *
 * \author Programmer: Daniel Tschupp - tud1@bfh.ch
 * \date Date: 09.08.19
 *
 * \details This Class is an oct-processing algorithm implementation that uses Cuda kernels for the processing. It
 * implements multiple streams working in a pipelined manner. It is conform to the I_Oct_Processing interface.
 */
class Oct_Cuda_Processing : public I_Oct_Processing
{
public:
    Oct_Cuda_Processing();
    ~Oct_Cuda_Processing() override;

    // Implementing interface
/** \brief This method calculates all the processing steps to get from a measured B-Scan
     * o the desired analyzed signal.
     *
     * @param       dataJunk	data structure containing the input and output buffers for data
    */
    virtual OctDataBufferPtr process_raw_data_to_bscan(OctDataBufferPtr dataJunk );

    /** \brief Method to set up the processing pipelines. This includes creating buffers on host as well as on
     * the devie. */
    void setProcessingConfiguration(const std::shared_ptr<Oct_Settings>& settings) override;

    /** @brief setDispersionCoeffs Method to set dispersion coefficients */
    void setDispersionCoeffs(const std::vector<float>& dispCoeffs) override;

    /** @brief Method to set background */
    void setBackground(const std::vector<float>& background) override;

    /** \brief Method to enable the background removal in the oct processing pipeline. */
    void enableSubBG( const bool& enable) override;

    /** \brief Method to enable the DC removal in the oct processing pipeline. */
    void enableRemoveDC( const bool& enable ) override;

    /** \brief Method to enable the spectral shaping in the oct processing pipeline. */
    void enableSpectralShapping( const bool& enable ) override;

    /** \brief Method to enable the remaping in the oct processing pipeline. */
    void enableRemapVector( const bool& enable ) override;

    /** \brief Method to enable the windowing in the oct processing pipeline. */
    void enableWindowing( const bool& enable ) override;

    /** \brief Method to enable the dispersion compensation in the oct processing pipeline. */
    void enableDispersionCompensation( const bool& enable ) override;

    /** \brief Method to enable the FFT in the oct processing pipeline. */
    void enableFFT( const bool& enable ) override;

    /** \brief Method to enable the Litte-Big Endian conversion in the oct processing pipeline. */
    void enableLittleBigEndian( const bool& enable ) override;

    /** \brief Static method that returns a vector with the information about all the available Cuda devices. */
    static std::vector<ProcInfo> getProcessingOptions(void);
private:
    bool process_raw_data_to_bscan(std::vector<unsigned short>& data, std::vector<float>& result );
    bool initBuffers();
    void initConvInterp();

    std::mutex m_GPU_Mutex;
    bool m_isInitialized;
    float m_OverSamplingRation;
    float m_dispCoeff_a2;
    float m_dispCoeff_a3;
    float m_dispCoeff_a4;
    int m_pmin;
    int m_pmax;
    int m_ProcessingOptions;
    size_t m_nbrSamplesCropped;
    size_t m_zeroPaddingFactor;
    std::shared_ptr<ConvolutionResampling> m_convolutionResampling;
    std::vector<float> m_inverseBackground;
    std::vector<float> m_remap;
    std::vector<int> m_convInds;
    std::vector<float> m_convCoeff;
    std::vector<unsigned int> m_smStartInds;
    std::vector<float> m_convCorrCoeff;
    size_t m_convBorder;

    // CUDA data
    OctGPUParams m_gpuParams;
    std::vector<OctIOdata> m_ioDatas;
    unsigned int m_actualStream;
    unsigned int m_streamMask;
};


#endif // OCT_CUDA_PROCESSING_H

/* Copyright details:
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
** Contact: Daniel Tschupp ( daniel.tschupp@gmail.com )
*/

