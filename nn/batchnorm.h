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
    Tensor<float> gamma; // shape (kernel_h * kernel_w)
    Tensor<float> beta;
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

    void load(string fileName, string gammaName, string betaName) {
        Loader<float> loader;
        loader.setFileName(fileName);
        loader.setItemName(gammaName);
        loader.load(gamma);
        loader.setItemName(betaName);
        loader.load(beta);
    }

    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.getDim() == 4);
        assert(input.getShape()[1] != 0); // number of input channels should be non-zero

        output = input;
        int N = input.getShape()[0];
        int C = input.getShape()[1];
        int H = input.getShape()[2];
        int W = input.getShape()[3];

        for (int c = 0; c < C; c++) {
            /* get mean */
            float sum = 0;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        sum += input(n, c, h, w);
                    }
                }
            }
            float mean = sum / (N * H * W);

            /* get variance */
            sum = 0;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        sum += pow(input(n, c, h, w) - mean, 2);
                    }
                }
            }

            float variance = sum / (N * H * W);
            /* set output */
            for (int n = 0; n < N; n++) {
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