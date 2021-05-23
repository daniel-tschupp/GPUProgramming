##############################################
# CUDA Snippet :Matrix Multiplication with shared Memory Snippet
#
# Author: Daniel Tschupp, tud1@bfh.ch
#
# March. 2019
##############################################

# Set Name of project and language
project(99_compareMatMulKernels LANGUAGES CXX)

# Set cmake version
cmake_minimum_required(VERSION 3.10)

# Dependencies
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED )

# Setup CUDA ********************************************************
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30) # Can't be enabled when using dynamic parallism
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)

set(CUDA_SEPARABLE_COMPILATION ON)

# Create executable using the specified source files
cuda_add_executable(${PROJECT_NAME}
    main.cu
    cuda_devicehelper.cpp
    kernels.cu
    )

# Link used libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})

# Define required c++ standard to C++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# Output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# Copy Files from Source Directory to bin directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/matrices
    DESTINATION ${CMAKE_BINARY_DIR}/bin/ )

# Set compile options, enable warnings
target_compile_options(${PROJECT_NAME} PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall>
$<$<CXX_COMPILER_ID:MSVC>: /W4>
)
