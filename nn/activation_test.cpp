#include "activation.h"
#include <iostream>

using namespace std;

Tensor<float> input;
input.init(2, 2, 3, 5);
float input_data[60]{
    -0.0865,  0.0194, -0.7198, -1.3064,  0.4679, -0.5583, -0.5058, -1.2496,
    -0.1417,  1.7925, -0.5095, -0.6528,  0.9676,  0.2013, -1.4495, -0.1064,
    -1.7182, -1.0932,  1.2530,  0.2698, -1.4022, -0.5537,  2.0462,  0.2437,
        1.2503, -1.7572,  0.5026,  1.6914, -0.3388,  0.6981, -0.7136,  0.6665,
        0.6714,  0.7490,  0.4754,  0.0725, -1.0816,  1.3718, -0.6054,  1.2531,
        0.2663, -0.7511, -0.8578, -0.6528, -0.0890, -0.8227, -0.8683, -0.5724,
        0.1978,  1.0399, -2.0217,  1.9208,  0.7178, -1.7369, -0.0659, -0.3337,
    -1.2640, -0.9317, -0.7178, -1.0136};
input.setData(input_data);

void test_ReLU() {
    cout << "==== test ReLU ====" << endl;    

    ReLU relu;
    Tensor<float> relu_output;
    relu.forward(input, relu_output);

    int dim = relu_output.getDim();
    cout << "relu_output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << relu_output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> relu_expected;
    relu_expected.init(2, 2, 3, 5);
    float relu_expected_data[60]{
        0.0000, 0.0194, 0.0000, 0.0000, 0.4679, 0.0000, 0.0000, 0.0000, 0.0000,
        1.7925, 0.0000, 0.0000, 0.9676, 0.2013, 0.0000, 0.0000, 0.0000, 0.0000,
        1.2530, 0.2698, 0.0000, 0.0000, 2.0462, 0.2437, 1.2503, 0.0000, 0.5026,
        1.6914, 0.0000, 0.6981, 0.0000, 0.6665, 0.6714, 0.7490, 0.4754, 0.0725,
        0.0000, 1.3718, 0.0000, 1.2531, 0.2663, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.1978, 1.0399, 0.0000, 1.9208, 0.7178, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    relu_expected.setData(relu_expected_data);
    cout << "same as relu_expected: " << relu_expected.isSame(relu_output) << endl;

    cout << "==== test LeakyReLU ====" << endl;    

    LeakyReLU leaky_relu;
    Tensor<float> leaky_relu_output;
    leaky_relu.forward(input, leaky_relu_output);

    int dim = leaky_relu_output.getDim();
    cout << "leaky_relu_output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << leaky_relu_output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> leaky_relu_expected;
    leaky_relu_expected.init(2, 2, 3, 5);
    float leaky_relu_expected_data[60]{
        -8.6532e-04,  1.9357e-02, -7.1983e-03, -1.3064e-02,  4.6791e-01,
        -5.5834e-03, -5.0583e-03, -1.2496e-02, -1.4173e-03,  1.7925e+00,
        -5.0955e-03, -6.5281e-03,  9.6763e-01,  2.0132e-01, -1.4495e-02,
        -1.0641e-03, -1.7182e-02, -1.0932e-02,  1.2530e+00,  2.6980e-01,
        -1.4022e-02, -5.5365e-03,  2.0462e+00,  2.4371e-01,  1.2503e+00,
        -1.7572e-02,  5.0264e-01,  1.6914e+00, -3.3885e-03,  6.9812e-01,
        -7.1365e-03,  6.6652e-01,  6.7138e-01,  7.4900e-01,  4.7539e-01,
         7.2474e-02, -1.0816e-02,  1.3718e+00, -6.0543e-03,  1.2531e+00,
         2.6630e-01, -7.5115e-03, -8.5776e-03, -6.5282e-03, -8.9006e-04,
        -8.2273e-03, -8.6830e-03, -5.7236e-03,  1.9778e-01,  1.0399e+00,
        -2.0217e-02,  1.9208e+00,  7.1785e-01, -1.7369e-02, -6.5902e-04,
        -3.3370e-03, -1.2640e-02, -9.3169e-03, -7.1777e-03, -1.0136e-02};
    leaky_relu_expected.setData(leaky_relu_expected_data);
    cout << "same as leaky_relu_expected: " << leaky_relu_expected.isSame(leaky_relu_output) << endl;
    
    cout << "==== test Sigmoid ====" << endl;    

    Sigmoid sigmoid;
    Tensor<float> sigmoid_output;
    sigmoid.forward(input, sigmoid_output);

    int dim = sigmoid_output.getDim();
    cout << "sigmoid_output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << sigmoid_output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> sigmoid_expected;
    sigmoid_expected.init(2, 2, 3, 5);
    float sigmoid_expected_data[60]{
        0.4784, 0.5048, 0.3274, 0.2131, 0.6149, 0.3639, 0.3762, 0.2228, 0.4646,
        0.8572, 0.3753, 0.3424, 0.7246, 0.5502, 0.1901, 0.4734, 0.1521, 0.2510,
        0.7778, 0.5670, 0.1975, 0.3650, 0.8856, 0.5606, 0.7773, 0.1471, 0.6231,
        0.8444, 0.4161, 0.6678, 0.3288, 0.6607, 0.6618, 0.6790, 0.6167, 0.5181,
        0.2532, 0.7977, 0.3531, 0.7778, 0.5662, 0.3206, 0.2978, 0.3424, 0.4778,
        0.3052, 0.2956, 0.3607, 0.5493, 0.7388, 0.1169, 0.8722, 0.6721, 0.1497,
        0.4835, 0.4173, 0.2203, 0.2826, 0.3279, 0.2663};
    sigmoid_expected.setData(sigmoid_expected_data);
    cout << "same as sigmoid_expected: " << sigmoid_expected.isSame(sigmoid_output) << endl;
}

int main() {
    test_ReLU();
}