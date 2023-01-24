#include "module.h"
#include <algorithm>
#include <cmath>

// save output for backward
class ReLU : public Module {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::max(*it, 0.f);
        }
    };
    void backward(){};
};

class LeakyReLU : public Module {
private:
    const float nagativeRate = 0.01;

public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::max(*it, 0.f) + nagativeRate * std::min(*it, 0.f);
        }
    };
    void backward(){};
};

class Sigmoid : public Module {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = 1 / (1 + std::exp(-*it));
            ;
        }
    };
    void backward(){};
};

class Tanh : public Module {
public:
    void forward(Tensor<float> &input, Tensor<float> &output) {
        output = input;
        float *it = output.begin();
        while (it) {
            *it = std::tanh(*it);
            ;
        }
    };
    void backward(){};
};

class Softmax : public Module {
private:
    const int dim = 1;

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