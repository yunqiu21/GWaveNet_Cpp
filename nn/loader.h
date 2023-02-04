#ifndef LOADER_H
#define LOADER_H

#include <fstream>
#include <iostream>
#include <string>
#include "tensor.h"
#include "../utils/json/json.h"
using namespace std;

template <typename T>
class Loader {
protected:   
    string fileName;
    string itemName;

public:
    void load(Tensor<T>& output);
    /* set names */
    void setFileName(string s);
    void setItemName(string s);
};

template <typename T>
void Loader<T>::setFileName(string s) {
    fileName = s;
};

template <typename T>
void Loader<T>::setItemName(string s) {
    itemName = s;
};

template <typename T>
void Loader<T>::load(Tensor<T>& output) {   
    Json::Value values;
    std::ifstream value_file(fileName, std::ifstream::binary);
    value_file >> values;

    int dim = values[itemName]["shape"].size();
    int *shape = new int[dim];
    for (int i = 0; i < dim; i++) {
        shape[i] = values[itemName]["shape"][i].asInt();
    }
    output.init(shape, dim);   
    float *it = output.begin();
    if (dim == 1) {       
        for (int k = 0; k < shape[0]; k++) {            
            *it = values[itemName]["__ndarray__"][k].asFloat();
            output.next(it);            
        }       
    } else if (dim == 2) {       
        for (int k = 0; k < shape[0]; k++) {
            for (int l = 0; l < shape[1]; l++) {
                *it = values[itemName]["__ndarray__"][k][l].asFloat();
                output.next(it);
            }
        }       
    } else if (dim == 3) {
        for (int j = 0; j < shape[0]; j++) {
            for (int k = 0; k < shape[1]; k++) {
                for (int l = 0; l < shape[2]; l++) {
                    *it = values[itemName]["__ndarray__"][j][k][l].asFloat();
                    output.next(it);
                }
            }
        }       
    } else if (dim == 4) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        *it = values[itemName]["__ndarray__"][i][j][k][l].asFloat();
                        output.next(it);
                    }
                }
            }
        }
    }
    delete[] shape;
};

#endif