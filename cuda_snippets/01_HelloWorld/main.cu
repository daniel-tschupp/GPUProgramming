#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cufft.h>

using namespace std;

__global__ void cuda_hello(char* cpu_D)
{
    cpu_D[0] = 'G';
}

int main()
{
    /***** Output Hello World from CPU ***********************************************/
    // Creating basic Hello World string
    string hello = "Hello World from ";

    // Creating a mutable char array for CPU
    char cpu_H[4] = "CPU";

    // Output the Greatings from the CPU first
    cout << hello << cpu_H << endl;

    /***** Output Hello World from GPU ***********************************************/
    // Creating the pointer to the char array on the GPU RAM and a size variable
    char* cpu_D;
    int size_cpu_D = 4 * sizeof(char);

    // Allocate memory on GPU RAM
    cudaMalloc((void**) &cpu_D, size_cpu_D);

    // Copy array from Host RAM to GPU RAM
    cudaMemcpy(cpu_D,cpu_H,size_cpu_D,cudaMemcpyHostToDevice);

    // Excecute Kernel (replace CPU with GPU)
    dim3 dimBlock(1,1); // Describse the number of Threads inside a Block.
    dim3 dimGrid(1,1);  // Describse the number of Blocks used in the Grid.
    cuda_hello<<<dimGrid, dimBlock>>>(cpu_D);

    // Read back the calculated data array
    cudaMemcpy(cpu_H, cpu_D, size_cpu_D, cudaMemcpyDeviceToHost);

    // Free Memory on GPU RAM
    cudaFree(cpu_D);

    // Output new GPU altered string
    cout << hello << cpu_H << endl;

    return 0;
}
