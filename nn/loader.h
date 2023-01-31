#ifndef LOADER_H
#define LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <json/json.h>
#include "tensor.h"

using namespace std;

template <typename T>
class Loader : public Tensor<T> {
protected:   
    string fileName;
    string itemName;

public:
    void load();
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
void Loader<T>::load() {   
    Json::Value values;
    std::ifstream value_file(fileName, std::ifstream::binary);
    value_file >> values;

    this->dim = values[itemName]["shape"].size();
    int *shape = new int[this->dim];
    this->dataCount = 1;
    for (int i = 0; i < this->dim; i++) {
        shape[i] = values[itemName]["shape"][i].asInt();
        this->dataCount *= shape[i];
    }
    this->shape = shape;
    this->data = new T[this->dataCount];
    
    float *it = this->begin();
    if (this->dim == 1) {       
        for (int k = 0; k < this->shape[0]; k++) {            
            *it = values[itemName]["__ndarray__"][k].asFloat();
            this->next(it);            
        }       
    } else if (this->dim == 2) {       
        for (int k = 0; k < this->shape[0]; k++) {
            for (int l = 0; l < this->shape[1]; l++) {
                *it = values[itemName]["__ndarray__"][k][l].asFloat();
                this->next(it);
            }
        }
       
    } else if (this->dim == 3) {
        for (int j = 0; j < this->shape[0]; j++) {
            for (int k = 0; k < this->shape[1]; k++) {
                for (int l = 0; l < this->shape[2]; l++) {
                    *it = values[itemName]["__ndarray__"][j][k][l].asFloat();
                    this->next(it);
                }
            }
        }       
    } else if (this->dim == 4) {
        for (int i = 0; i < this->shape[0]; i++) {
            for (int j = 0; j < this->shape[1]; j++) {
                for (int k = 0; k < this->shape[2]; k++) {
                    for (int l = 0; l < this->shape[3]; l++) {
                        *it = values[itemName]["__ndarray__"][i][j][k][l].asFloat();
                        this->next(it);
                    }
                }
            }
        }
    }
};

#endif