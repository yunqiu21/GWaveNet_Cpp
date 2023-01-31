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

    // /* getters */
    // int getDim();
    // const int *const getShape();
    // T* getData() { return (T*)data; }

    /* set names */
    void setFileName(string s);
    void setItemName(string s);
};

// template <typename T>
// Loader<T>::Loader() {
//     data = nullptr;
//     shape = nullptr;
//     dim = 0;
// };

template <typename T>
void Loader<T>::setFileName(string s) {
    fileName = s;
};

template <typename T>
void Loader<T>::setItemName(string s) {
    itemName = s;
};

// template <typename T>
// void Loader<T>::init(int row, int column) {
//     dim = 2;
//     shape = new int[2]{row, column};
//     dataCount = row * column;
//     data = new T[dataCount];
// };

// template <typename T>
// void Loader<T>::init(int C, int H, int W) {
//     dim = 3;
//     shape = new int[3]{C, H, W};
//     dataCount = C * H * W;
//     data = new T[dataCount];
// };

// template <typename T>
// void Loader<T>::init(int N, int C, int H, int W) {
//     dim = 4;
//     shape = new int[4]{N, C, H, W};
//     dataCount = N * C * H * W;
//     data = new T[dataCount];
// };

template <typename T>
void Loader<T>::load() {   
    Json::Value values;
    std::ifstream value_file(fileName, std::ifstream::binary);
    value_file >> values;

    this->dim = values[itemName]["shape"].size();
    cout << this->dim << endl;
    int *shape = new int[this->dim];
    this->dataCount = 1;
    for (int i = 0; i < this->dim; i++) {
        shape[i] = values[itemName]["shape"][i].asInt();
        this->dataCount *= shape[i];
    }
    this->shape = shape;
    this->data = new T[this->dataCount];
    
    // data = new T[dataCount];
    float *it = this->begin();
    if (this->dim == 1) {       
        // data = new T[shape[0]][shape[1]];
        for (int k = 0; k < this->shape[0]; k++) {            
            *it = values[itemName]["__ndarray__"][k].asFloat();
            cout << *it << " ";
            this->next(it);            
        }       
    } else if (this->dim == 2) {       
        // data = new T[shape[0]][shape[1]];
        for (int k = 0; k < this->shape[0]; k++) {
            for (int l = 0; l < this->shape[1]; l++) {
                *it = values[itemName]["__ndarray__"][k][l].asFloat();
                cout << *it << " ";
                this->next(it);
            }
        }
       
    } else if (this->dim == 3) {
        // data = new T[shape[0]][shape[1]][shape[2]];
        for (int j = 0; j < this->shape[0]; j++) {
            for (int k = 0; k < this->shape[1]; k++) {
                for (int l = 0; l < this->shape[2]; l++) {
                    *it = values[itemName]["__ndarray__"][j][k][l].asFloat();
                    cout << *it << " ";
                    this->next(it);
                }
            }
        }
       
    } else if (this->dim == 4) {
        // data = new T[shape[0]][shape[1]][shape[2]][shape[3]];
        for (int i = 0; i < this->shape[0]; i++) {
            for (int j = 0; j < this->shape[1]; j++) {
                for (int k = 0; k < this->shape[2]; k++) {
                    for (int l = 0; l < this->shape[3]; l++) {
                        *it = values[itemName]["__ndarray__"][i][j][k][l].asFloat();
                        cout << *it << " ";
                        this->next(it);
                    }
                }
            }
        }
    }
};

// template <typename T>
// int Loader<T>::getDim() {
//     return dim;
// }

// template <typename T>
// const int *const Loader<T>::getShape() {
//     return shape;
// }



// void LoadData(float data[32][32][1][2], int shape[4], string fileName, string itemName)
// {   
//     shape[0] = 32;
//     shape[1] = 32;
//     shape[2] = 1;
//     shape[3] = 2;
//     Json::Value values;
//     std::ifstream value_file("data/metr_epoch_33_2.8.json", std::ifstream::binary);
//     value_file >> values;

//     for (int i = 0; i < 32; i++) {
//         for (int j = 0; j < 32; j++) {
//             for (int k = 0; k < 1; k++) {
//                 for (int l = 0; l < 2; l++) {
//                     data[i][j][k][l] = values[itemName]["__ndarray__"][i][j][k][l].asFloat();
//                     std::cout << data[i][j][k][l] << " ";
//                 }
//             }
//         }
//     }
// }