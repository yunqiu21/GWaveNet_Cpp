#ifndef GCN_H
#define GCN_H

#include "list.h"
#include "nconv.h"
#include "nn/activation.h"
#include "nn/conv.h"
#include "nn/loader.h"
#include <iostream>

using namespace std;

class GCN {
private:
    NConv nconv;
    int c_in;
    Conv2d mlp;
    Tensor<float> mlp_weight;
    Tensor<float> mlp_bias;
    int order;
    Dropout dropout;

    void concat(Tensor<float> &x, Tensor<float> &out, int &cIdx) {
        for (int n = 0; n < x.getShape()[0]; n++) {
            for (int c = 0; c < x.getShape()[1]; c++) {
                for (int h = 0; h < x.getShape()[2]; h++) {
                    for (int w = 0; w < x.getShape()[3]; w++) {
                        out(n, cIdx + c, h, w) = x(n, c, h, w);
                    }
                }
            }
        }
        cIdx += x.getShape()[1];
    };

public:
    GCN(){};

    GCN(int c_in, int c_out, float dropout = 0.3, int support_len = 3, int order = 2)
        : c_in((order * support_len + 1) * c_in), order(order), mlp(Conv2d(this->c_in, c_out, 1, 1)),
          dropout(Dropout(dropout)){};

    void load(string fileName, string weightName, string biasName) {
        Loader<float> loader;
        loader.setFileName(fileName);
        loader.setItemName(weightName);
        loader.load(mlp_weight);
        loader.setItemName(biasName);
        loader.load(mlp_bias);
    }

    void forward(Tensor<float> &x, List<Tensor<float>> &support, Tensor<float> &output) {
        Tensor<float> out;
        out.init(x.getShape()[0], x.getShape()[1] * (1 + support.size() * order), x.getShape()[2], x.getShape()[3]);

        int cIdx = 0;
        concat(x, out, cIdx);
        for (int i = 0; i < support.size(); i++) {
            Tensor<float> x1;
            nconv.forward(x, support(i), x1);
            concat(x1, out, cIdx);
            for (int j = 2; j < order + 1; j++) {
                Tensor<float> x2;
                nconv.forward(x1, support(i), x2);
                concat(x2, out, cIdx);
                x1 = x2;
            }
        }

        if (mlp_weight.isInit() && mlp_bias.isInit()) {
            mlp.setWeight(mlp_weight);
            mlp.setBias(mlp_bias);
        }

        Tensor<float> out1;
        mlp.forward(out, out1);
        dropout.forward(out1, output);
    }
};

#endif