#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>

#include <cuda.h>
#include <cufft.h>

using namespace std;

void readMatrix(float* const pM, const int size_M, const string filename);
void createMatrix(float* const pM, const int size_M, const string filename);
void showMatrix(const float* const pM, const int columns, const int rows);

void readMatrix(float* const pM, const int size_M, const string filename){
    ifstream inputData(filename);
    for(int i = 0; i<size_M; i++){
        string tmp;
        inputData >> tmp;
        pM[i] = std::stof(tmp);
    }
}

void createMatrix(float* const pM, const int size_M, const string filename){
    srand((unsigned)time(0));
    for(int i= 0; i < size_M; i++)
        pM[i] = static_cast <float> (rand()) / RAND_MAX * static_cast <float>(10);

    ofstream arrayData(filename); // File Creation(on C drive)

    for(int k=0;k<size_M;k++)
    {
        arrayData<<pM[k]<<endl;
    }
    cout << "Created File: " << filename << " with " << size_M << " Elements." << endl;
}

void showMatrix(const float* const pM, const int columns, const int rows){
    cout << "Matrix:" << endl;

    for(int i = 0; i < rows; i++){
        for(int k = 0; k < columns; k++)
            cout << pM[columns*i + k] << "\t";
        cout << endl;
    }
}

int main(){
    // Describe matrix to create
    const int columns = 512;
    const int rows = 512;
    const string filename = "Matrix_512x512_B.txt";
    // create array and insert random values and write array to text file.
    float matrix[columns*rows];cout << "Test" << endl;
    createMatrix(matrix, columns*rows, filename);
    // create array and read text file to store values inside the array.
    float inputArray[columns*rows];
    readMatrix(inputArray, columns*rows, filename);

    // display the matrix in the console.
    showMatrix(inputArray, columns, rows);

    return 0;
}



