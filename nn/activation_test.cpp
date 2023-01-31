#include "activation.h"
#include <iostream>

using namespace std;

Tensor<float> input; /* input for all tests except softmax */

void test_ReLU() {
    cout << "==== test ReLU ====" << endl;    

    ReLU relu;
    Tensor<float> relu_output;
    relu.forward(input, relu_output);

    int relu_dim = relu_output.getDim();
    cout << "relu_output dimension: " << relu_dim << ", shape: ";
    for (int i = 0; i < relu_dim; i++) {
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
}

void test_LeakyReLU() {
    cout << "==== test LeakyReLU ====" << endl;    

    LeakyReLU leaky_relu;
    Tensor<float> leaky_relu_output;
    leaky_relu.forward(input, leaky_relu_output);

    int leaky_relu_dim = leaky_relu_output.getDim();
    cout << "leaky_relu_output dimension: " << leaky_relu_dim << ", shape: ";
    for (int i = 0; i < leaky_relu_dim; i++) {
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
}

void test_Sigmoid() {
    cout << "==== test Sigmoid ====" << endl;    

    Sigmoid sigmoid;
    Tensor<float> sigmoid_output;
    sigmoid.forward(input, sigmoid_output);

    int sigmoid_dim = sigmoid_output.getDim();
    cout << "sigmoid_output dimension: " << sigmoid_dim << ", shape: ";
    for (int i = 0; i < sigmoid_dim; i++) {
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

void test_Tanh() {
    cout << "==== test Tanh ====" << endl;    

    Tanh tanh;
    Tensor<float> tanh_output;
    tanh.forward(input, tanh_output);

    int tanh_dim = tanh_output.getDim();
    cout << "tanh_output dimension: " << tanh_dim << ", shape: ";
    for (int i = 0; i < tanh_dim; i++) {
        cout << tanh_output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> tanh_expected;
    tanh_expected.init(2, 2, 3, 5);
    float tanh_expected_data[60]{
        -0.0863,  0.0194, -0.6168, -0.8634,  0.4365, -0.5067, -0.4667, -0.8482,
        -0.1408,  0.9460, -0.4696, -0.5736,  0.7476,  0.1986, -0.8956, -0.1060,
        -0.9376, -0.7980,  0.8491,  0.2634, -0.8858, -0.5033,  0.9672,  0.2390,
         0.8484, -0.9422,  0.4642,  0.9343, -0.3264,  0.6032, -0.6129,  0.5827,
         0.5859,  0.6346,  0.4426,  0.0724, -0.7938,  0.8791, -0.5409,  0.8492,
         0.2602, -0.6358, -0.6951, -0.5736, -0.0888, -0.6765, -0.7005, -0.5171,
         0.1953,  0.7778, -0.9655,  0.9580,  0.6155, -0.9399, -0.0658, -0.3218,
        -0.8522, -0.7314, -0.6155, -0.7672};
    tanh_expected.setData(tanh_expected_data);
    cout << "same as tanh_expected: " << tanh_expected.isSame(tanh_output) << endl;
}

void test_Softmax() {
    cout << "==== test Softmax ====" << endl;    

    Tensor<float> softmax_input;
    softmax_input.init(4, 5);
    float softmax_input_data[20]{
         0.0674, -1.0838,  0.2145, -1.2188, -1.2034,  2.4634,  0.7295, -0.8012,
        -0.0117,  1.2978, -0.3526, -1.5425, -0.7073,  1.3479, -1.0193,  0.2188,
         2.8235, -1.2821, -0.9532, -0.0842};
    input.setData(softmax_input_data);

    Softmax softmax;
    Tensor<float> softmax_output;
    softmax.forward(softmax_input, softmax_output);

    int softmax_dim = softmax_output.getDim();
    cout << "softmax_output dimension: " << softmax_dim << ", shape: ";
    for (int i = 0; i < softmax_dim; i++) {
        cout << softmax_output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> softmax_expected;
    softmax_expected.init(4, 5);
    float softmax_expected_data[20]{
        0.3299, 0.1043, 0.3821, 0.0911, 0.0926, 0.6208, 0.1096, 0.0237, 0.0522,
        0.1935, 0.1251, 0.0381, 0.0877, 0.6850, 0.0642, 0.0633, 0.8562, 0.0141,
        0.0196, 0.0468};
    softmax_expected.setData(softmax_expected_data);
    cout << "same as softmax_expected: " << softmax_expected.isSame(softmax_output) << endl;
}

int main() {
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
    
    test_ReLU();
    test_LeakyReLU();
    test_Sigmoid();
    test_Tanh();
    test_Softmax();
}
