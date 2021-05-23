# GPU Programming with CUDA

This Repo includes a brief documentation that explains the main topics when programming GPUs with CUDA. It furthermore contains example snippets of kernels and how to launch them for teaching perpose.

At last it includes an GPU Acceleration Project for the OCT image processing.

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

### Add extend PWD
Add the following lines .profile file if they don't already exist:

if [ -d "$HOME/bin" ] ; then

    PATH="$HOME/bin:$PATH"

fi


if [ -d "$HOME/.local/bin" ] ; then

    PATH="$HOME/.local/bin:$PATH"

fi

if [-d "/usr/local/cuda-10.1/bin" ] ; then

	PATH="/usr/local/cuda-10.1/bin:$PATH"

	PATH="/usr/local/cuda-10.1/bin/includes:$PATH"

fi


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

