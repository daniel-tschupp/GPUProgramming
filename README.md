# GPU accelerated OCT algorithms

As my first project in my master studies I successfully accelerated the oct processing chain on the GPU. A processing speed
of 2.6GS/s was reached.

This repository contains the final report in a pdf form as well as the CUDA code and the C++ code using it.
It can be found in the oct_algorithm_implementation folder.

Further there are some cuda sample snippets that demonstrate the different techniques used in the project.

If you want to run the code, please ensure the following requirements:

## Prerequisits
The following prerequisits are mandatory to run the code in this repo:
*	g++-6
*	cmake (version3.10 or higher)
*	cuda installation from NVIDIA website: [Toolkit NVIDIA](https://developer.nvidia.com/cuda-downloads)
*	Depending on the cuda version one needs specific graphics driver version in order for cuda to work properly. (see below)
*	It may be neccessary to add cuda to PWD

### CUDA and Driver overview
[<img src="https://gitlab.ti.bfh.ch/optoLab/gpu_algorithms/raw/master/installSettings/CudaToolVersionDriverOverview.png">]()

Installing specific drivers using the PPA:
*	sudo add-apt-repository ppa:graphics-drivers/ppa
*	sudo apt update
*	sudo apt install nvidia-x
	x: version to install
*	Rebooting
*	Test installation via nvidia-settings (if it work --> dirver works)

More infos can be found here: [How to install Nvidia Drivers](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)

## Performance Analysis

To Analyze the performance of a GPU accelerated software nvidia provides tools that come with the default installation. Be aware: To use those tools on linux root access is required.

A software can be analzed with the following command in sudo su mode:

nvprof -o outputFileName.nvprof ./executable

The -o option specifies an output file containing the results. The .nvprof ending is required in order to import the data with the nvvp (nvidia visual profiler)which is the graphical user interface.

When importing (not opening) the xxx.nvprof file one need to specifie if multiple parallel streams were used.

## Snippet Collection Overview

List of Sample snippets:
*	01_HelloWorld
*	02_CreateRandomMatrixFile
*	03_matricCalc
*	04_SystemInfo
*	05_matricCalcBlocks
*	06_matricCalcBlocksSM
*	07_constantMemory
*	08_optimSubtraction
*	09_MemoryPaging
*	10_SimpleAsyncStream
*	11_AsyncStreamClass
*	12_AsyncStreamClassImproved
*	13_ProperCudaErrorHandling


More information about those snippets can be found inside the documentaion in chapter 2.

## OCT Image Processing

*	Documentation describing the algorithm
*	Cuda Implementation of the algorithm 

