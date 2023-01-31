#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "tensor.h"
#include "loader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>

using namespace std;

class BatchNorm2D {
private:
    Loader<float> gamma; // shape (kernel_h * kernel_w)
    Loader<float> beta;
    int channels;
    float eps = 0.00005;
    float momentum = 0.1;

public:
    BatchNorm2D(){};
    BatchNorm2D(int channels) : channels(channels) {
        gamma.init(channels);
        beta.init(channels);

        float array[channels];
        memset(array, 0.f, channels * sizeof(float));
        gamma.setData(array);

        memset(array, 1.f, channels * sizeof(float));
        beta.setData(array);
    };

    void loadGamma(string fileName, string itemName) {
        gamma.setFileName(fileName);
        gamma.setItemName(itemName);
        gamma.load();
        int gamma_dim = gamma.getDim();
        cout << "gamma dimension: " << gamma_dim << ", shape: ";
        for (int i = 0; i < gamma_dim; i++) {
            cout << gamma.getShape()[i] << " ";
        }
        cout << endl;
    }

    void loadBeta(string fileName, string itemName) {
        beta.setFileName(fileName);
        beta.setItemName(itemName);
        beta.load();
        int beta_dim = beta.getDim();
        cout << "beta dimension: " << beta_dim << ", shape: ";
        for (int i = 0; i < beta_dim; i++) {
            cout << beta.getShape()[i] << " ";
        }
        cout << endl;
    }

    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.getDim() == 4);
        assert(input.getShape()[1] != 0); // number of input channels should be non-zero

        output = input;
        int N = input.getShape()[0];
        int C = input.getShape()[1];
        int H = input.getShape()[2];
        int W = input.getShape()[3];

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                /* get mean */
                float sum = 0;
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        sum += input(n, c, h, w);
                    }
                }
                float mean = sum / (H * W);
                cout << "Mean: " << mean << endl;

                /* get variance */
                sum = 0;
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        sum += pow(input(n, c, h, w) - mean, 2);
                    }
                }
                cout << "sum: " << sum << endl;
                float variance = sum / (H * W);
                cout << "Var:  " << variance << endl;
                /* set output */
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        output(n, c, h, w) = (output(n, c, h, w) - mean) / sqrt(variance + eps) * gamma(c) + beta(c);
                    }
                }
            }
        }
    };

    void backward(){};
};

#endif