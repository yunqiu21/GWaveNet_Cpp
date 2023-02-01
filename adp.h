#ifndef ADP_H
#define ADP_H

#include "nn/activation.h"
#include "nn/matmul.h"
#include "nn/loader.h"
#include <random>
#include <iostream>

using namespace std;

class Adp {
private:
    Tensor<float> nodevec1;
    Tensor<float> nodevec2;
    ReLU relu;
    Softmax softmax;

public:
    Adp(int num_nodes) {
        nodevec1.init(num_nodes, 10);
        nodevec2.init(10, num_nodes);
    }

    void randomInit() {
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);

        float *it = nodevec1.begin();
        while (it) {
            *it = distribution(generator);
            nodevec1.next(it);
        }

        it = nodevec2.begin();
        while (it) {
            *it = distribution(generator);
            nodevec2.next(it);
        }
    };

    void load(string fileName, string nodevec1Name, string nodevec2Name) {
        Loader<float> loader;
        loader.setFileName(fileName);
        loader.setItemName(nodevec1Name);
        loader.load(nodevec1);
        loader.setItemName(nodevec2Name);
        loader.load(nodevec2);
    }

    void forward(Tensor<float> &output) {
        Tensor<float> out1;
        matmul2D(nodevec1, nodevec2, out1);
        Tensor<float> out2;
        relu.forward(out1, out2);
        softmax.forward(out2, output);
    }
};

#endif