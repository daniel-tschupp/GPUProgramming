cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "/usr/bin/gcc-8")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")

project(testCudaLib VERSION 1.0.0 LANGUAGES CXX)


add_subdirectory(lib)

add_executable(${PROJECT_NAME} main.cpp )

target_link_libraries(${PROJECT_NAME} PUBLIC octCudaAlgorithmLib)
