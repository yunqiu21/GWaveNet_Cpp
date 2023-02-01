#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <random>

// save output for backward
class ReLU {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::max(*it, 0.f);
            output.next(it);
        }
    };
    void backward(){};
};

class LeakyReLU {
private:
    float nagativeRate = 0.01;

public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::max(*it, 0.f) + nagativeRate * std::min(*it, 0.f);
            output.next(it);
        }
    };
    void backward(){};
};

class Sigmoid {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = 1 / (1 + std::exp(-*it));
            output.next(it);
        }
    };
    void backward(){};
};

class Tanh {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::tanh(*it);
            output.next(it);
        }
    };
    void backward(){};
};

class Softmax {
private:
    int dim = 1;

public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.getDim() == 2);
        output = input;

        const int *shape = input.getShape();
        for (int i = 0; i < shape[0]; i++) {
            float sum = 0;
            for (int j = 0; j < shape[1]; j++) {
                sum += exp(output(i, j));
            }

            if (sum != 0) {
                for (int j = 0; j < shape[1]; j++) {
                    output(i, j) = exp(output(i, j)) / sum;
                }
            }
        }
    };
    void backward(){};
};

class Dropout {
private:
    float p;

public:
    Dropout(float p = 0.3) : p(p) {}

    void forward(Tensor<float> &input, Tensor<float> &output, bool training = false) {
        // should do nothing when not training
        output = input;
        if (training) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::bernoulli_distribution d(p);
            float *it = output.begin();
            while (it) {
                if (d(gen))
                    *it = 0;
                output.next(it);
            }
        }
    };
    void backward(){};
};

#endif