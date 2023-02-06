#include "tensor.h"
#include <iostream>
#include <cassert>

using namespace std;

void test1() {
    cout << "==== test tensor ====" << endl;
    Tensor<int> t;
    assert(!t.isInit());
    assert(!t.getShape());

    t.init(3, 2, 2);
    int num = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                t(i, j, k) = num;
                num++;
            }
        }
    }

    cout << "tensor   value: ";
    int *val = t.begin();
    while (val) {
        cout << *val << " ";
        t.next(val);
    }
    cout << endl;

    Tensor<int> t2;
    t2 = t;
    cout << "tensor 2 value: ";
    val = t2.begin();
    while (val) {
        cout << *val << " ";
        t2.next(val);
    }
    cout << endl;
}

void test2() {
    cout << "==== test padLastDim ====" << endl;
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

    int receptive_field = 6;
    assert(input.getDim() == 4);

    if (input.getShape()[3] < receptive_field) {
        input.padLastDim(receptive_field);
    }

    input.padLastDim(6);

    int dim = input.getDim();
    cout << "padded input dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << input.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 3, 6);
    float expected_data[72]{
         0.0000, -0.0865,  0.0194, -0.7198, -1.3064,  0.4679,  0.0000, -0.5583,
        -0.5058, -1.2496, -0.1417,  1.7925,  0.0000, -0.5095, -0.6528,  0.9676,
         0.2013, -1.4495,  0.0000, -0.1064, -1.7182, -1.0932,  1.2530,  0.2698,
         0.0000, -1.4022, -0.5537,  2.0462,  0.2437,  1.2503,  0.0000, -1.7572,
         0.5026,  1.6914, -0.3388,  0.6981,  0.0000, -0.7136,  0.6665,  0.6714,
         0.7490,  0.4754,  0.0000,  0.0725, -1.0816,  1.3718, -0.6054,  1.2531,
         0.0000,  0.2663, -0.7511, -0.8578, -0.6528, -0.0890,  0.0000, -0.8227,
        -0.8683, -0.5724,  0.1978,  1.0399,  0.0000, -2.0217,  1.9208,  0.7178,
        -1.7369, -0.0659,  0.0000, -0.3337, -1.2640, -0.9317, -0.7178, -1.0136};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(input) << endl;
}

int main() {
    test1();
    test2();
}