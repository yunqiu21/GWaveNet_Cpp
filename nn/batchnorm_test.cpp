#include "loader.h"
#include "batchnorm.h"
#include <iostream>

using namespace std;

void test1() {
    cout << "==== test 1 (1,2,1,2) ====" << endl;  

    Tensor<float> input;    
    input.init(1, 2, 1, 2);
    float input_data[4]{
        -0.3572,  0.2753, -0.4522, -0.7150};
    input.setData(input_data);

    BatchNorm2D bn;
    bn.load("../data/test_data.json", "bn.weight", "bn.bias");

    Tensor<float> output;
    bn.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(1, 2, 1, 2);
    float expected_data[4]{
         2.0005, 5.9995, 7.9957, 2.0043};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

void test2() {
    cout << "==== test 2 (1,2,2,2) ====" << endl;  

    Tensor<float> input;
    input.init(1, 2, 2, 2);
    float input_data[8]{
        1.0594, -0.2994,  0.9339, -0.9727,  2.1594,  0.1584,  0.7436,  0.3791};
    input.setData(input_data);

    BatchNorm2D bn;
    bn.load("../data/test_data.json", "bn.weight", "bn.bias");

    Tensor<float> output;
    bn.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;    
    expected.init(1, 2, 2, 2);
    float expected_data[8]{
         6.0647,  2.8733,  5.7700,  1.2920, 10.0053,  2.2967,  4.5511,  3.1469};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

void test3() {
    cout << "==== test 3 (2,2,3,5) ====" << endl;  

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

    BatchNorm2D bn;
    bn.load("../data/test_data.json", "bn.weight", "bn.bias");

    Tensor<float> output;
    bn.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 3, 5);
    float expected_data[60]{
         4.0293,  4.2869,  2.4887,  1.0617,  5.3780,  2.8816,  3.0093,  1.1999,
         3.8950,  8.6003,  3.0003,  2.6517,  6.5936,  4.7294,  0.7136,  5.2039,
         0.9074,  2.5734,  8.8277,  6.2068,  1.7497,  4.0116, 10.9421,  6.1372,
         8.8205,  0.8034,  6.8273,  9.9963,  4.5844,  7.3485,  2.5038,  5.8611,
         5.8730,  6.0618,  5.3962,  4.4161,  1.6086,  7.5769,  2.7670,  7.2881,
         4.8876,  2.4126,  2.1530,  2.6517,  4.0232,  3.2945,  3.1729,  3.9617,
         6.0148,  8.2596,  0.0983, 10.6078,  7.4010,  0.8575,  5.3119,  4.5980,
         2.1181,  3.0039,  3.5741,  2.7856};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test1();
    test2();
    test3();
}

