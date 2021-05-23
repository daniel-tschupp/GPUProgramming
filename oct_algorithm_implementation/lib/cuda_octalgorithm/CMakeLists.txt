##############################################
# OCTPROLIGHTGPU : Creates library to process
# oct raw data on nvidia GPU's
#
# Usage:
#
# find_package(cudaOctAlgorithm REQUIRED)
# target_link_libraries(targetName PUBLIC cudaOctAlgorithm::cudaOctAlgorithm)
# or
# add_subdirectory(cudaOctAlgorithm)
#
#
# Author: Patrik Arnold, aop2@bfh.ch
#
# Feb. 2019
##############################################

cmake_minimum_required(VERSION 3.10)
project(cudaOctAlgorithm VERSION 1.0.0 LANGUAGES CXX)

##############################################
# Declare dependencies
find_package(CUDA REQUIRED)

# Setup CUDA ################################
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)

set(CUDA_SEPARABLE_COMPILATION ON)

##############################################
# Create target and set properties
CUDA_ADD_LIBRARY(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/src/cuda_processoctdata.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/cuda_utils.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/cuda_octkernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_processoctdata.h
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_includes.h
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_utils.h
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_errorchecking.h
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_types.h
    ${CMAKE_CURRENT_LIST_DIR}/include/cuda_octkernels.h
    )

# Add cufft
CUDA_ADD_CUFFT_TO_TARGET( ${PROJECT_NAME} )

#Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

option(BUILD_SHARED_LIBS "Build shared library" OFF)
# Link Cuda libs
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})

IF( CMAKE_BUILD_TYPE MATCHES "DEBUG" )
target_compile_definitions(${PROJECT_NAME} PUBLIC MODULE_CUDA_DEBUG CUDA_DEBUG_MSG)
ENDIF( CMAKE_BUILD_TYPE MATCHES "DEBUG" )
IF( CMAKE_BUILD_TYPE MATCHES "Debug" )
target_compile_definitions(${PROJECT_NAME} PUBLIC MODULE_CUDA_DEBUG CUDA_DEBUG_MSG)
ENDIF( CMAKE_BUILD_TYPE MATCHES "Debug" )


#Set target properties
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

# Output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# Automatically create a module definition (.def) file with all global symbols
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

# Set compile options, enable warnings
target_compile_options(${PROJECT_NAME} PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall>
$<$<CXX_COMPILER_ID:MSVC>: /W3>
)

##############################################
# Installation instructions

# 'make install' to the correct locations (provided by GNUInstallDirs).
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in into 'cmake' folder
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)
