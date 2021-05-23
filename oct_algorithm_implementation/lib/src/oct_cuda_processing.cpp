/*
** Copyright (c) 2020 Daniel Tschupp, details see at the end of the document.
*/

#include "oct_cuda_processing.h"
#include "oct_convolutionresampling.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "cuda_processoctdata.h"
#include <assert.h>

#define MODULE_CUDA_DEBUG
#ifdef MODULE_CUDA_DEBUG
#include "cuda_profiler_api.h"
#endif

Oct_Cuda_Processing::Oct_Cuda_Processing():
    m_isInitialized(false),
    m_actualStream(0),
    m_streamMask(3)
{
#ifdef MODULE_CUDA_DEBUG
    cudaProfilerStart();
#endif
    m_ProcessingOptions = static_cast<int>(ProcessingOptions::ALL);
    //m_gpuParams.POptions = static_cast<int>(ProcessingOptions::ALL);
    // Check for devices
    int nDevices{0};
    cudaGetDeviceCount( &nDevices );
    if(nDevices != 0)
    {
        m_ioDatas.clear();

        for(unsigned int i = 0; i <= m_streamMask; i++){
            m_ioDatas.push_back( OctIOdata() );
            m_ioDatas.at(i).stream = gpuGetNewStream();
        }
    }
}


Oct_Cuda_Processing::~Oct_Cuda_Processing()
{
    if( m_ioDatas.size() > 0)
    {
        if( m_isInitialized )
        {
            FreeCudaConfigBuffers(&m_gpuParams);

            for(size_t iStream = 0; iStream <= m_streamMask; iStream++){
                FreeCudaDataBuffers(&m_ioDatas.at(iStream));
            }
        }

        for(unsigned int i = 0; i < m_ioDatas.size(); i++){
            gpuCleanUpStream(m_ioDatas.at(i).stream);
        }
    }
#ifdef MODULE_CUDA_DEBUG
    cudaProfilerStop();
#endif
}

void Oct_Cuda_Processing::enableDispersionCompensation(const bool &enable )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::DISP_COMP) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::DISP_COMP);

    float dispCoeff[] = {m_dispCoeff_a2, m_dispCoeff_a3, m_dispCoeff_a4};
    if(m_isInitialized){
        writeCudaConstDispCoeff(dispCoeff, 3);
    }
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableRemoveDC(const bool &enable )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::DC_REMOVE) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::DC_REMOVE);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);

}

void Oct_Cuda_Processing::enableLittleBigEndian( const bool& enable ){
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::CONVERT_LITTLE_BIG_ENDIAN) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::CONVERT_LITTLE_BIG_ENDIAN);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableRemapVector(const bool &enable )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::REMAP) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::REMAP);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableWindowing(const bool &enable )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::WINDOWING) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::WINDOWING);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableSpectralShapping(const bool &enable )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::SPECTRAL_SHAPPING) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::SPECTRAL_SHAPPING);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableFFT(const bool &enable)
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::FFT) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::FFT);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

void Oct_Cuda_Processing::enableSubBG(const bool &enable)
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    enable  ?   m_ProcessingOptions |= static_cast<int>(ProcessingOptions::BG_REMOVE) :
                m_ProcessingOptions &= ~static_cast<int>(ProcessingOptions::BG_REMOVE);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
}

OctDataBufferPtr Oct_Cuda_Processing::process_raw_data_to_bscan( OctDataBufferPtr dataJunk )
{
    const std::lock_guard<std::mutex> lock(m_GPU_Mutex);
    if(process_raw_data_to_bscan( dataJunk->m_raw, dataJunk->m_processed )){
        dataJunk->nbrSamplesAScanProc = m_gpuParams.nbrOfSamplesAScanRaw*m_zeroPaddingFactor;
        if(IS_REMAP_ACTIVE(m_ProcessingOptions))
            dataJunk->nbrSamplesAScanProc = dataJunk->nbrSamplesAScanProc * m_OverSamplingRation;
        return dataJunk;
    }
    else
        return nullptr;
}


void Oct_Cuda_Processing::setProcessingConfiguration(const std::shared_ptr<Oct_Settings>& settings)
{
    // reset
    if( m_isInitialized )
    {
        FreeCudaConfigBuffers(&m_gpuParams);

        for(size_t iStream = 0; iStream <= m_streamMask; iStream++){
            FreeCudaDataBuffers(&m_ioDatas.at(iStream));
        }
        m_isInitialized = false;
    }

    // Check Cuda devices
    int nDevices{0};
    cudaGetDeviceCount( &nDevices );

    // Oversampling Ratio
    m_OverSamplingRation = settings->getOverSamplingRatio();

    // Zeropadding
    m_zeroPaddingFactor = settings->getZeroPaddingFactor();

    // Gerneral Scan information
    m_gpuParams.nbrOfSamplesAScanRaw = settings->getNbrSamplesAScan();
    m_gpuParams.overSamplingRatio = m_OverSamplingRation;
    m_gpuParams.zeroPaddingFactor = m_zeroPaddingFactor;

    m_nbrSamplesCropped = settings->getNbrPixelCropped();
    m_gpuParams.nbrOfAScans = settings->getNbrAscansInBscan();

    // Remap and Background vector
    m_remap = settings->getRemapVector();
    m_inverseBackground = settings->getBackgroundVector();
    m_gpuParams.bgMean = settings->getMeanBackground();

    // Region of interest
    m_pmax = static_cast<int>(settings->getPMax());
    m_pmin = static_cast<int>(settings->getPMin());

    // Disperion
    m_dispCoeff_a2 = settings->getDispersionCoeffA2();
    m_dispCoeff_a3 = settings->getDispersionCoeffA3();
    m_dispCoeff_a4 = settings->getDispersionCoeffA4();

    this->enableFFT(settings->getDoFFT());
    this->enableSubBG(settings->getDoReference() || settings->getDoDark() || settings->getDoSample());
    this->enableRemoveDC(settings->getDoRemoveDC());
    this->enableWindowing(settings->getDoWindow());
    this->enableRemapVector(settings->getDoRemap());
    this->enableSpectralShapping(settings->getDoSpectralShaping());
    this->enableDispersionCompensation(settings->getDoDispComp());
    this->enableLittleBigEndian(settings->getDoConvertEndian());

    initBuffers();
}

void Oct_Cuda_Processing::setDispersionCoeffs(const std::vector<float> &dispCoeffs)
{
    float dispCoeff[] = {dispCoeffs[0], dispCoeffs[1], dispCoeffs[2]};
    writeCudaConstDispCoeff(dispCoeff, 3);
}

void Oct_Cuda_Processing::setBackground(const std::vector<float> &background)
{
    m_inverseBackground = background;
    writeCudaBuffer(m_gpuParams.baGround_C, m_inverseBackground.data(), m_gpuParams.nbrOfSamplesAScanRaw * sizeof(float));
}

vector<ProcInfo> Oct_Cuda_Processing::getProcessingOptions(void){
    std::vector<ProcInfo> list;

    // Check for devices
    int nDevices{0};
    cudaGetDeviceCount( &nDevices );
    if(nDevices <= 0)
    {
        return std::vector<ProcInfo>();
    }

    for(int i=0; i<nDevices;i++){
        cudaDeviceProp deviceProp;
        CudaSafeAPICall(cudaGetDeviceProperties(&deviceProp, i));
        std::stringstream ss;
        ss << "CUDA: " << deviceProp.name;
        ProcInfo info = { ss.str(), ProcessingType::CUDA, 0, 0};
        list.push_back(info);
    }
    return list;
}

bool Oct_Cuda_Processing::process_raw_data_to_bscan(std::vector<unsigned short>& data, std::vector<float>& result )
{
    if( !m_isInitialized )
    {
        printf("Oct_Processing: Not initialized\n");
        return false;
    }
/*
 *
 *
 *  if(m_useCUDA)
+    {
+        // CUDA Processing Main Interface
+        m_ioDatas.at(m_actualStream & m_streamMask).srcBscan = data.data();
+        m_ioDatas.at(m_actualStream & m_streamMask).dstResult = result.data();
+        cudaStream_t* resStream = nullptr;
+        if(m_actualStream >= m_streamMask){
+            resStream = m_ioDatas.at(m_actualStream-1 & m_streamMask).stream;
+        }
+        ProcessGPU(m_gpuParams, m_ioDatas.at(m_actualStream & m_streamMask), resStream);
+        m_actualStream++;
+        return true;
+    }
+    // default
+    std::cout << "Oct_Processing: No processing possible\n";

 *
 * */
    // CUDA Processing Main Interface
#ifdef DEBUG_GLOB_ALLOC_MEM
    std::cout << "Source data vector length: " << data.size()/m_gpuParams.nbrOfAScans << std::endl;
    std::cout << "Result data vector length: " << result.size()/m_gpuParams.nbrOfAScans << std::endl;
#endif
    m_ioDatas.at(m_actualStream & m_streamMask).srcBscan = data.data();
    m_ioDatas.at(m_actualStream & m_streamMask).dstResult = result.data();

    ProcessGPU(m_gpuParams, m_ioDatas.at(m_actualStream & m_streamMask), m_ProcessingOptions);
    m_actualStream++;

    return true;
}

bool Oct_Cuda_Processing::initBuffers()
{
    initConvInterp();

    // Allocate CUDA Buffers
#ifdef DEBUG_GLOB_ALLOC_MEM
    printf("Initialize Config Buffers.\n");
#endif
    InitConfigCudaBuffers(&m_gpuParams);
    for(unsigned int iStream = 0; iStream<=m_streamMask; iStream++){
#ifdef DEBUG_GLOB_ALLOC_MEM
        printf("Initialize Data Buffer of Stream: %d.\n", iStream);
#endif
        InitDataCudaBuffers(&m_ioDatas.at(iStream), m_gpuParams);
    }
    float dispCoeff[] = {m_dispCoeff_a2, m_dispCoeff_a3, m_dispCoeff_a4};
    int sizeSMInds = m_smStartInds.size();

    writeCudaConstSizeSMStartInds(&sizeSMInds);
    writeCudaConstSMStartInds(m_smStartInds.data(), m_smStartInds.size());
    writeCudaConstDispCoeff(dispCoeff, 3);
    writeCudaConstWindowPmin(&m_pmin);
    writeCudaConstWindowPmax(&m_pmax);
    writeCudaConstOverSamplingRatio(&m_OverSamplingRation);
    writeCudaConstProcessingOptions(&m_ProcessingOptions);
    writeCudaBuffer(m_gpuParams.conCn_C,  m_convolutionResampling->getCn().data(),  m_gpuParams.nbrOfSamplesAScanRaw*m_gpuParams.overSamplingRatio*m_gpuParams.zeroPaddingFactor*  sizeof(float));
    writeCudaBuffer(m_gpuParams.baGround_C, m_inverseBackground.data(), m_gpuParams.nbrOfSamplesAScanRaw * sizeof(float));
    writeCudaBuffer(m_gpuParams.convCoeff_C, m_convCoeff.data(), m_gpuParams.nbrOfSamplesAScanRaw * m_gpuParams.overSamplingRatio * m_gpuParams.kernelSize * sizeof(float));
    writeCudaBuffer(m_gpuParams.convInds_C, m_convInds.data(), m_gpuParams.nbrOfSamplesAScanRaw * m_gpuParams.overSamplingRatio * m_gpuParams.kernelSize * sizeof(int));

    m_isInitialized = true;
    return m_isInitialized;
}

void Oct_Cuda_Processing::initConvInterp()
{
    m_convolutionResampling.reset(new ConvolutionResampling(m_remap,
                                                            m_gpuParams.nbrOfSamplesAScanRaw,
                                                            m_gpuParams.nbrOfSamplesAScanRaw*m_OverSamplingRation,
                                                            m_gpuParams.nbrOfSamplesAScanRaw*m_OverSamplingRation*m_zeroPaddingFactor,
                                                            m_OverSamplingRation, m_pmin, m_pmax));
    m_gpuParams.kernelSize = m_convolutionResampling->getKernelSize();

    // Create vectors with Coefficients (C) and Indices (N_ind) used for Convolution Interp.
    m_convCoeff.clear();
    m_convInds.clear();
    m_smStartInds.clear();

    m_convInds.push_back(-1);
    m_convInds.push_back(-1);
    m_convInds.push_back(-1);

    m_convCoeff.push_back(0.0f);
    m_convCoeff.push_back(0.0f);
    m_convCoeff.push_back(0.0f);


    m_convBorder = m_convolutionResampling->getBorder();
    for (size_t k = m_convBorder; k < m_gpuParams.nbrOfSamplesAScanRaw*m_OverSamplingRation-m_convBorder; ++k)
    {
        for (size_t j = 0; j < m_gpuParams.kernelSize; ++j)
        {
            size_t ind = k * m_gpuParams.kernelSize + j;
            int N_ind =  m_convolutionResampling->getInds()[ind];

            if(N_ind == -1)
            {
                m_convInds.push_back(-1);   // NOTE: set valid indices
                m_convCoeff.push_back(0.0f); // NOTE: set C to zero to avoid check in convolution
            }
            else
            {
                m_convInds.push_back( N_ind );
                m_convCoeff.push_back(m_convolutionResampling->getC()[ind]);
            }
        }
    }

    m_convInds.push_back(-1);
    m_convInds.push_back(-1);
    m_convInds.push_back(-1);

    m_convCoeff.push_back(0.0f);
    m_convCoeff.push_back(0.0f);
    m_convCoeff.push_back(0.0f);

    // init correction coefficients (Cn)
    m_convCorrCoeff.clear();
    for(size_t j = 0; j != m_gpuParams.nbrOfSamplesAScanRaw*m_OverSamplingRation*m_zeroPaddingFactor; j++)
    {
        m_convCorrCoeff.push_back( m_convolutionResampling->getCn()[j] );
    }

    // As the remap isn't linear it's not possible to input partition the data for gpu processing. Therefor a vector is needed that defines which input data junk
    // a gpu batch processing needs to calculate the corresponding ouput datas. In the following code segment a batch size aka tile width is defined and the starting
    // indecees are read out for those tiles.
    unsigned int tileWidth = 256;
    m_smStartInds.push_back(0);
    for(int iInds = (tileWidth*m_gpuParams.kernelSize); iInds < m_convInds.size(); iInds+=(tileWidth*m_gpuParams.kernelSize)){
        unsigned int smallestIndsOfActKernel = m_gpuParams.nbrOfSamplesAScanRaw;
        for(int ikernel = 0; ikernel < m_gpuParams.kernelSize; ikernel++){
            int actVal = m_convInds.at(iInds + ikernel);
            if(actVal != -1 && actVal < smallestIndsOfActKernel)
                smallestIndsOfActKernel = actVal;
        }
        m_smStartInds.push_back(smallestIndsOfActKernel);
    }

    // Check how much the biggest data junk for a tile is to detemine the data padding into the shared memory on the gpu
    int remapPad = 0;
    for(int i = 0; i < m_smStartInds.size()-1; i++){
        if(remapPad < std::abs((int)(m_smStartInds.at(i+1)-m_smStartInds.at(i)))){
            remapPad = std::abs((int)(m_smStartInds.at(i+1)-m_smStartInds.at(i)));
        }
    }
    remapPad -= tileWidth;
    m_gpuParams.tileWidth = tileWidth;
}

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

