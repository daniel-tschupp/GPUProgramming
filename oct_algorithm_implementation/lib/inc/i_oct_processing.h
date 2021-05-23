/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#ifndef I_OCT_PROCESSING_H
#define I_OCT_PROCESSING_H

#include <string>
#include <memory>


/**  \brief EInterpolation  : Enumeration with the diferent remap interpolation algorithms for cpu processing only. */
enum class EInterpolation:int{ ECubicSpline, EConvolution };

/** \brief ProcessingOptions    : Enumeration with the different processing steps that can be enabled. */
enum class ProcessingOptions:int { ALL=0xFF7F, DISP_COMP=1, WINDOWING=2, REMAP=4, DC_REMOVE=8, BG_REMOVE=16, FFT=32, SPECTRAL_SHAPPING=64, CONVERT_LITTLE_BIG_ENDIAN=128 };

/** \brief ProcessingType   : Enumeration with the different processing algorithm options. */
enum class ProcessingType:int { SingleCPU, MultiCPU, CUDA, OPENCL };

/** \brief ProcInfo : Container datatype stroring infos of a processing unit. */
struct ProcInfo
{
    std::string name;
    ProcessingType type;
    unsigned int platformID;
    unsigned int deviceID;
};

class OctDataBuffer
{
public:
	std::vector<unsigned short> m_raw;
	std::vector<float> m_processed;
	size_t nbrSamplesAScanProc;
};
typedef std::shared_ptr<OctDataBuffer> OctDataBufferPtr;
class Oct_Settings
{
public:
	virtual float getOverSamplingRatio() = 0;
	virtual float getZeroPaddingFactor() = 0;
	virtual float getNbrSamplesAScan() = 0;
	virtual unsigned int getNbrPixelCropped() = 0;
	virtual unsigned int getNbrAscansInBscan() = 0;
	virtual std::vector<float> getRemapVector() = 0;
	virtual std::vector<float> getBackgroundVector() = 0;
	virtual float getMeanBackground() = 0;
	virtual size_t getPMax() = 0;
	virtual size_t getPMin() = 0;
	virtual float getDispersionCoeffA2() = 0;
	virtual float getDispersionCoeffA3() = 0;
	virtual float getDispersionCoeffA4() = 0;
	virtual bool getDoFFT() = 0;
	virtual bool getDoReference() = 0;
	virtual bool getDoDark() = 0;
	virtual bool getDoSample() = 0;
	virtual bool getDoRemoveDC() = 0;
	virtual bool getDoWindow() = 0;
	virtual bool getDoRemap() = 0;
	virtual bool getDoSpectralShaping() = 0;
	virtual bool getDoDispComp() = 0;
	virtual bool getDoConvertEndian() = 0;
};
/**
 * \class I_Oct_Processing
 *
 * \author Programmer: Daniel Tschupp - tud1@bfh.ch
 * \date Date: 13.08.19
 *
 * Description:
 * This Interface class provides the means to uniformly use different approaches for the oct
 * processing depending on the available hardware.
 */
class I_Oct_Processing{
public:

    virtual ~I_Oct_Processing(){}

    /** \brief This method calculates all the processing steps to get from a measured B-Scan the desired analyzed signal.
     *
     * @param       dataJunk       	A class containing to data buffers. One with the input
     *                              values and one with the result values.
     */
    virtual OctDataBufferPtr process_raw_data_to_bscan(OctDataBufferPtr dataJunk) = 0;
    /** \brief Method to set up the processing pipelines. This includes creating buffers on host as well as on
     * the devie. */
    virtual void setProcessingConfiguration(const std::shared_ptr<Oct_Settings>& settings) = 0;

    /** @brief Method to set dispersion coefficients */
    virtual void setDispersionCoeffs(const std::vector<float>& dispCoeffs) = 0;

    /** @brief Method to set background */
    virtual void setBackground(const std::vector<float>& background) = 0;

    /** \brief Method to enable the background removal in the oct processing pipeline. */
    virtual void enableSubBG(const bool &enable) = 0;

    /** \brief Method to enable the DC removal in the oct processing pipeline. */
    virtual void enableRemoveDC(const bool &enable) = 0;

    /** \brief Method to enable the spectral shaping in the oct processing pipeline. */
    virtual void enableSpectralShapping(const bool &enable) = 0;

    /** \brief Method to enable the remaping in the oct processing pipeline. */
    virtual void enableRemapVector(const bool &enable) = 0;

    /** \brief Method to enable the windowing in the oct processing pipeline. */
    virtual void enableWindowing(const bool &enable) = 0;

    /** \brief Method to enable the dispersion compensation in the oct processing pipeline. */
    virtual void enableDispersionCompensation(const bool &enable) = 0;

    /** \brief Method to enable the FFT in the oct processing pipeline. */
    virtual void enableFFT( const bool& enable ) = 0;

    /** \brief Method to enable the Litte-Big Endian conversion in the oct processing pipeline. */
    virtual void enableLittleBigEndian( const bool& enable ) = 0;
};

#endif // I_OCT_PROCESSING_H

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

