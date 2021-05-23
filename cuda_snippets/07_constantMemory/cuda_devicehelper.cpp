#include "cuda_devicehelper.h"

CUDA_DeviceHelper::CUDA_DeviceHelper(int DeviceID, bool print):
    mDeviceID(DeviceID)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    if(mDeviceID >= dev_count){
        cout << "Device with ID: " << DeviceID << "doesn't exist!";
    }
    else{
        cudaGetDeviceProperties(&mDevProps, mDeviceID);
        if(print)
            this->printDeviceData();
    }
    mNumWarpSheduler = 0;
    if(mDevProps.major == 1 || mDevProps.major == 2)
        mNumWarpSheduler = 1;
    if(mDevProps.major == 3)
        mNumWarpSheduler = 6;
    if(mDevProps.major == 6 && mDevProps.minor == 0)
        mNumWarpSheduler = 2;
    if(mDevProps.major == 6 && mDevProps.minor == 0)
        mNumWarpSheduler = 4;
    if(mDevProps.major == 7)
        mNumWarpSheduler = 2;

    if(mDevProps.major < 3)
        mResidentBlocksInSM = 8;
    if(mDevProps.major < 3)
            mResidentBlocksInSM = 8;
    if(mDevProps.major >= 3 && mDevProps.minor < 5)
            mResidentBlocksInSM = 16;
    if(mDevProps.major >= 5){
        mResidentBlocksInSM = 32;
        if(mDevProps.major == 7 && mDevProps.minor == 5)
            mResidentBlocksInSM = 16;
    }

}

const cudaDeviceProp& CUDA_DeviceHelper::getDevProps(void) const{
    return mDevProps;
}
int CUDA_DeviceHelper::getNumberOfBurstRequestsPerWarpSheduler(int TileWidth) const{
    return ceil(TileWidth/(float)mDevProps.warpSize);
}
int CUDA_DeviceHelper::calcParallelGlobMemRequest(int TileWidth)const{
    return this->getNumberOfSMs() * this->getNumberOfCuncurrentWarpsPerSM() * this->getNumberOfBurstRequestsPerWarpSheduler(TileWidth);
}
void CUDA_DeviceHelper::checkDimConfigs(dim3 gridDim, dim3 blockDim) const{
    int gridVol = gridDim.x*gridDim.y*gridDim.z;
    int blockVol = blockDim.x*blockDim.y*blockDim.z;
    cout << endl << "*********************** Computation Analysis ************************************" << endl << endl;
    cout << "Block Analysis: " << endl;
    cout << "Number of resident Blocks allowed per SM: " << mResidentBlocksInSM << endl;
    int blocksAssignedSM = gridVol/mDevProps.multiProcessorCount;
    int actualBlocksPerSM = blocksAssignedSM;
    cout << "Number of Blocks assigned to SM: " << blocksAssignedSM << endl;
    int blocksDueToSharedMemoryLimit = mDevProps.sharedMemPerBlock / blockVol;
    if(blocksDueToSharedMemoryLimit < actualBlocksPerSM) actualBlocksPerSM = blocksDueToSharedMemoryLimit;
    cout << "Number of Blocks due to shared memory limitation: " << blocksDueToSharedMemoryLimit << endl;
    int blocksDueToSMThreadsLimit = mDevProps.maxThreadsPerMultiProcessor / blockVol;
    if(blocksDueToSMThreadsLimit < actualBlocksPerSM) actualBlocksPerSM = blocksDueToSMThreadsLimit;
    cout << "Number of Blocks due to SM-Threads limitation: " << blocksDueToSMThreadsLimit << endl;
    cout << "Max Number of Registers allowed in Kernel: " << mDevProps.regsPerMultiprocessor / blockVol << endl << endl;

    cout << "Thread Analysis: " << endl;
    cout << "Number of Threads in Block allowed: " << mDevProps.maxThreadsPerBlock << endl;
    cout << "Number of Threads assigned to Block: " << blockVol << endl;
    cout << "Number of Threads allowed in SM: " << mDevProps.maxThreadsPerMultiProcessor << endl;
    cout << "Number of Threads assigned to SM: " << actualBlocksPerSM * blockVol << endl << endl;

    cout << "Warp Analysis: " << endl;
    cout << "Number of concurrent Warps possible: " << mNumWarpSheduler << endl;
    cout << "Number of assigned Warps: " << (actualBlocksPerSM * blockVol)/(float)mDevProps.warpSize << endl;
    cout << "Ratio of assigned Warps to residing Warps: " << ((actualBlocksPerSM * blockVol)/(float)mDevProps.warpSize) / (float)mNumWarpSheduler << endl << endl;

    cout << "Global Memory Analysis: " << endl;
    cout << "Parallel memory burst requests: " << this->calcParallelGlobMemRequest(blockDim.x) << endl << endl;

    cout << "Data Analysis: " << endl;
    cout << "Grid Size: " << gridVol << endl;
    cout << "Square Grid Width: " << ceil(sqrt(gridVol)) << endl;
    cout << "Total number of Threads: " << gridVol * blockVol << endl;
    cout << "Block Size: " << blockVol << endl;
    cout << "Square Block Width: " << ceil(sqrt(blockVol)) << endl;
    cout << "Burst width if Tile = Block: " << blockDim.x << endl << endl;

    if(blockVol <= mDevProps.maxThreadsPerBlock && actualBlocksPerSM * blockVol <= mDevProps.maxThreadsPerMultiProcessor)
        cout << "----> Code will probably run" << endl << endl;
    else
        cout << "----> There's an error..." << endl << endl;
}
void CUDA_DeviceHelper::printDeviceData(void) const{
    cout << "Printing Cuda Device Properties: " << endl;
    cout << "name: " << mDevProps.name << endl << endl;
    cout << "concurrentKernels: " << mDevProps.concurrentKernels << endl;
    cout << "multiProcessorCount: " << mDevProps.multiProcessorCount << endl;
    cout << "clockRate: " << mDevProps.clockRate/1000000.0f << "MHz" << endl;
    cout << "warpSize: " << mDevProps.warpSize << endl ;
    cout << "asyncEngineCount" << mDevProps.asyncEngineCount << endl << endl;

    cout << "totalGlobalMem: " << mDevProps.totalGlobalMem/1000000000.0f << "GB" << endl;
    cout << "sharedMemPerBlock: " << mDevProps.sharedMemPerBlock/1000.0f << "kB" << endl;
    cout << "totalConstMem: " << mDevProps.totalConstMem/1000.0f << "kB" << endl;
    cout << "regsPerBlock: " << mDevProps.regsPerBlock/1000.0f << "kB" << endl ;
    cout << "memoryBusWidth: " << mDevProps.memoryBusWidth << endl;
    cout << "memoryClockRate: " << mDevProps.memoryClockRate << endl;
    cout << "memPitch: " << mDevProps.memPitch << endl;
    cout << "pageableMemoryAccess: " << mDevProps.pageableMemoryAccess << endl;
    cout << "unifiedAddressing: " << mDevProps.unifiedAddressing << endl << endl;

    cout << "maxThreadsPerBlock: " << mDevProps.maxThreadsPerBlock << endl;
    cout << "maxThreadsPerSM: " << mDevProps.maxThreadsPerMultiProcessor << endl;
    cout << "maxThreadsDim[x]: " << mDevProps.maxThreadsDim[0] << endl;
    cout << "maxThreadsDim[y]: " << mDevProps.maxThreadsDim[1] << endl;
    cout << "maxThreadsDim[z]: " << mDevProps.maxThreadsDim[2] << endl;
    cout << "maxGridSize[x]: " << mDevProps.maxGridSize[0] << endl;
    cout << "maxGridSize[y]: " << mDevProps.maxGridSize[1] << endl;
    cout << "maxGridSize[z]: " << mDevProps.maxGridSize[2] << endl << endl;

    cout << "major: " << mDevProps.major << endl;
    cout << "minor: " << mDevProps.minor << endl;
    cout << "textureAlignment: " << mDevProps.textureAlignment << endl;
    cout << "deviceOverlap: " << mDevProps.deviceOverlap << endl;
    cout << "kernelExecTimeoutEnabled: " << mDevProps.kernelExecTimeoutEnabled << endl;
    cout << "integrated: " << mDevProps.integrated << endl;
    cout << "canMapHostMemory: " << mDevProps.canMapHostMemory << endl;
    cout << "computeMode: " << mDevProps.computeMode << endl;
    cout << "ECCEnabled: " << mDevProps.ECCEnabled << endl;
    cout << "pciBusID: " << mDevProps.pciBusID << endl;
    cout << "pciDeviceID: " << mDevProps.pciDeviceID << endl;
    cout << "tccDriver: " << mDevProps.tccDriver << endl << endl;
}
int CUDA_DeviceHelper::getGPUFrequency(void) const{
    return mDevProps.clockRate;
}
int CUDA_DeviceHelper::getNumberOfSMs(void) const{
    return mDevProps.multiProcessorCount;
}
int CUDA_DeviceHelper::getNumberOfConcurrentKernels(void) const{
    return mDevProps.concurrentKernels;
}
int CUDA_DeviceHelper::getWarpSize() const{
    return mDevProps.warpSize;
}
int CUDA_DeviceHelper::getNumberOfCuncurrentWarpsPerSM(void) const{
    return mNumWarpSheduler;
}
int CUDA_DeviceHelper::CUDA_DeviceHelper::getGlobMemorySize(void) const{
    return mDevProps.totalGlobalMem;
}
int CUDA_DeviceHelper::getSharedMemorySize(void) const{
    return mDevProps.sharedMemPerBlock;
}
int CUDA_DeviceHelper::getRegisterSize(int numUsedThreadsPerSM) const{
    return mDevProps.regsPerBlock / numUsedThreadsPerSM;
}
int CUDA_DeviceHelper::getMaxThreadsPerBlock(void) const{
    return mDevProps.maxThreadsPerBlock;
}
int CUDA_DeviceHelper::getMaxThreadsPerSM(void) const{
    return mDevProps.maxThreadsPerMultiProcessor;
}
string CUDA_DeviceHelper::getName(void) const{
    return mDevProps.name;
}
void CUDA_DeviceHelper::readMatrix(float* const pM, const int size_M, const string filename){
    ifstream inputData(filename);
    for(int i = 0; i<size_M; i++){
        string tmp;
        inputData >> tmp;
        pM[i] = std::stof(tmp);
    }
}
void CUDA_DeviceHelper::createMatrix(float* const pM, const int size_M, const string filename){
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

void CUDA_DeviceHelper::showMatrix(const float* const pM, const int columns, const int rows){
    cout << "Matrix:" << endl;

    for(int i = 0; i < rows; i++){
        for(int k = 0; k < columns; k++)
            cout << pM[columns*i + k] << "\t";
        cout << endl;
    }
}
