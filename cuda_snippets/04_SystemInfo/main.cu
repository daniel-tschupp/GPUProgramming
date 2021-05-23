#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cufft.h>

using namespace std;

int main()
{
    // Read out the number of graphical devices.
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cout << "Number of Devices: " << dev_count << endl;

    // Get Device Properties of each of those devices.
    cudaDeviceProp dev_prop;
    for(int i = 0; i < dev_count; i++){
        // Get Cuda Device Properties
        cudaGetDeviceProperties(&dev_prop, i);

        // Create storing file filepath
        string path = "../../AnalyzedDevices/";
        string filename = path + dev_prop.name + ".txt";
        ofstream devDataFile(filename); // File (on C drive)

        // Write properties to file
        devDataFile << "Device name: \t" << dev_prop.name << endl << endl;

        devDataFile << "Clock Rate: \t" << dev_prop.clockRate/1000000.0 << "MHz" << endl;
        devDataFile << "Number of SMs: \t" << dev_prop.multiProcessorCount << endl;
        devDataFile << "Number of Cuncurrent Kernels: \t" << dev_prop.concurrentKernels << endl;
        devDataFile << "Warp Size: \t" << dev_prop.warpSize << endl;
        devDataFile << "Total Global Memory: \t" << dev_prop.totalGlobalMem/1000000000.0 << " GBytes" << endl;
        devDataFile << "Total Constant Memory: \t" << dev_prop.totalConstMem/1000.0 << "kBytes" << endl << endl;

        devDataFile << "Shared Memory per Block: \t" << dev_prop.sharedMemPerBlock/1000.0 << " kBytes" << endl;
        devDataFile << "Memory Pitch: \t" << dev_prop.memPitch << endl;
        devDataFile << "Max Threads per Block: \t" << dev_prop.maxThreadsPerBlock << endl;
        devDataFile << "Max Threads per Block in X: \t" << dev_prop.maxThreadsDim[0] << endl;
        devDataFile << "Max Threads per Block in Y: \t" << dev_prop.maxThreadsDim[1] << endl;
        devDataFile << "Max Threads per Block in Z: \t" << dev_prop.maxThreadsDim[2] << endl << endl;

        devDataFile << "Max Blocks per Grid in X: \t" << dev_prop.maxGridSize[0] << endl;
        devDataFile << "Max Blocks per Grid in Y: \t" << dev_prop.maxGridSize[1] << endl;
        devDataFile << "Max Blocks per Grid in Z: \t" << dev_prop.maxGridSize[2] << endl;

        // Print properties in the console
        cout << "Device name: \t" << dev_prop.name << endl << endl;

        cout << "Clock Rate: \t" << dev_prop.clockRate/1000000.0 << "MHz" << endl;
        cout << "Number of SMs: \t" << dev_prop.multiProcessorCount << endl;
        cout << "Number of Cuncurrent Kernels: \t" << dev_prop.concurrentKernels << endl;
        cout << "Warp Size: \t" << dev_prop.warpSize << endl;
        cout << "Total Global Memory: \t" << dev_prop.totalGlobalMem/1000000000.0 << " GBytes" << endl;
        cout << "Total Constant Memory: \t" << dev_prop.totalConstMem/1000.0 << "kBytes" << endl << endl;

        cout << "Shared Memory per Block: \t" << dev_prop.sharedMemPerBlock/1000.0 << " kBytes" << endl;
        cout << "Memory Pitch: \t" << dev_prop.memPitch << endl;
        cout << "Max Threads per Block: \t" << dev_prop.maxThreadsPerBlock << endl;
        cout << "Max Threads per Block in X: \t" << dev_prop.maxThreadsDim[0] << endl;
        cout << "Max Threads per Block in Y: \t" << dev_prop.maxThreadsDim[1] << endl;
        cout << "Max Threads per Block in Z: \t" << dev_prop.maxThreadsDim[2] << endl << endl;

        cout << "Max Blocks per Grid in X: \t" << dev_prop.maxGridSize[0] << endl;
        cout << "Max Blocks per Grid in Y: \t" << dev_prop.maxGridSize[1] << endl;
        cout << "Max Blocks per Grid in Z: \t" << dev_prop.maxGridSize[2] << endl;
    }

    return 0;
}
