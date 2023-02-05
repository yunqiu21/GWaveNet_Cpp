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
    Tensor<float> gamma;
    Tensor<float> beta;
    Tensor<float> running_mean;
    Tensor<float> running_var;
    int channels;
    float eps = 1e-5;
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

    void load(string fileName, string gammaName, string betaName,
              string runningMean = "", string runningVar = "") {
        Loader<float> loader;
        loader.setFileName(fileName);
        loader.setItemName(gammaName);
        loader.load(gamma);
        loader.setItemName(betaName);
        loader.load(beta);
        if (runningMean == "" || runningVar == "") return;
        loader.setItemName(runningMean);
        loader.load(running_mean);
        loader.setItemName(runningVar);
        loader.load(running_var);
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
            float mean;
            if (running_mean.isInit()) {
                mean = running_mean(c);
            } else {
                float sum = 0;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            sum += input(n, c, h, w);
                        }
                    }
                }
                mean = sum / (N * H * W);
            }
            /* get variance */
            float variance;
            if (running_var.isInit()) {
                variance = running_var(c);
            } else {
                float sum = 0;
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            sum += pow(input(n, c, h, w) - mean, 2);
                        }
                    }
                }
                variance = sum / (N * H * W);
            }
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