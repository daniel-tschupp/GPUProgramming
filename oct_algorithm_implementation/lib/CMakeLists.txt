##############################################
# octLib : Provides the means to process OCT 
# A-Scans with a Nvidia GPU
#
# Usage:
#
# find_package(octCudaAlgorithmLib REQUIRED)
# target_link_libraries(targetName PUBLIC octCudaAlgorithmLib)
#
# Author: Daniel Tschupp, daniel.tschupp@gmail.com
#
# Nov. 2020
##############################################

cmake_minimum_required(VERSION 3.10)
project(octCudaAlgorithmLib VERSION 1.0.0 LANGUAGES CXX)

##############################################
# DEPENDENCIES
find_package(Qt5 REQUIRED COMPONENTS PrintSupport SerialPort)

##############################################
# Create target and set properties

add_library(${PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/i_oct_processing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/oct_cuda_processing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/oct_convolutionresampling.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/oct_convolutionresampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/oct_cuda_processing.cpp
    # ALGLIB     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/ap.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/alglibinternal.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/alglibmisc.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/linalg.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/specialfunctions.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/integration.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/interpolation.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/solvers.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/optimization.h     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/ap.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/alglibinternal.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/alglibmisc.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/linalg.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/specialfunctions.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/integration.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/interpolation.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/solvers.cpp     
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10/optimization.cpp
    )

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})


add_subdirectory(cuda_octalgorithm)
target_link_libraries(${PROJECT_NAME} PUBLIC cudaOctAlgorithm)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:inc>
    $<INSTALL_INTERFACE:alglib_3.10>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/alglib_3.10>
    )

set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

# Define required c++ features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_lambdas cxx_lambda_init_captures cxx_std_17)


##############################################
# Installation instructions

# 'make install' to the correct locations (provided by GNUInstallDirs).
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) # This is for Windows

# This makes the project importable from the install directory
# Put config file in into 'cmake' folder
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE cudaOctAlgorithm.cmake)
