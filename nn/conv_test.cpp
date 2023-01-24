#include "conv.h"
#include <iostream>

using namespace std;

void test1() {
    cout << "==== test1 ====" << endl;
    Tensor<float> input;
    input.init(2, 3, 4, 5);

    /* init input */
    int count = 1;
    float *i = input.getFirst();
    while (i) {
        *i = count;
        input.getNext(i);
        count++;
    }

    Conv2d conv = Conv2d(3, 2, 2, 2, 2, true);

    Tensor<float> weight;
    weight.init(2, 3, 2, 2);
    count = 1;
    i = weight.getFirst();
    while (i) {
        *i = count;
        weight.getNext(i);
        count++;
    }
    conv.setWeight(weight);

    Tensor<float> bias;
    bias.init(2);
    count = 1;
    i = bias.getFirst();
    while (i) {
        *i = count;
        bias.getNext(i);
        count++;
    }
    conv.setBias(bias);

    Tensor<float> output;
    conv.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 2, 3);
    float data[24]{2813, 2891, 2969, 3203, 3281, 3359, 6702, 6924, 7146,
                   7812, 8034, 8256, 7493, 7571, 7649, 7883, 7961, 8039,
                   20022, 20244, 20466, 21132, 21354, 21576};
    expected.setData(data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

void test2() {
    cout << "==== test2 ====" << endl;
    Tensor<float> input;
    input.init(2, 3, 4, 5);

    /* init input */
    int count = 1;
    float *i = input.getFirst();
    while (i) {
        *i = count;
        input.getNext(i);
        count++;
    }

    Conv2d conv = Conv2d(3, 2, 2, 2, 1, true);

    Tensor<float> weight;
    weight.init(2, 3, 2, 2);
    count = 1;
    i = weight.getFirst();
    while (i) {
        *i = count;
        weight.getNext(i);
        count++;
    }
    conv.setWeight(weight);

    Tensor<float> bias;
    bias.init(2);
    count = 1;
    i = bias.getFirst();
    while (i) {
        *i = count;
        bias.getNext(i);
        count++;
    }
    conv.setBias(bias);

    Tensor<float> output;
    conv.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 3, 4);
    float data[48]{2546., 2624., 2702., 2780., 2936., 3014., 3092., 3170., 3326.,
                   3404., 3482., 3560., 6003., 6225., 6447., 6669., 7113., 7335.,
                   7557., 7779., 8223., 8445., 8667., 8889., 7226., 7304., 7382.,
                   7460., 7616., 7694., 7772., 7850., 8006., 8084., 8162., 8240.,
                   19323., 19545., 19767., 19989., 20433., 20655., 20877., 21099., 21543.,
                   21765., 21987., 22209.};
    expected.setData(data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

void test3() {
    cout << "==== test3 ====" << endl;
    Tensor<float> input;
    input.init(2, 5, 2);

    /* init input */
    int count = 1;
    float *i = input.getFirst();
    while (i) {
        *i = count;
        input.getNext(i);
        count++;
    }

    Conv1d conv = Conv1d(5, 2, 2, 1, true);
    Tensor<float> weight;
    weight.init(2, 5, 2);
    count = 1;
    i = weight.getFirst();
    while (i) {
        *i = count;
        weight.getNext(i);
        count++;
    }
    conv.setWeight(weight);

    Tensor<float> bias;
    bias.init(2);
    count = 1;
    i = bias.getFirst();
    while (i) {
        *i = count;
        bias.getNext(i);
        count++;
    }
    conv.setBias(bias);
    Tensor<float> output;
    conv.forward(input, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 1);
    float data[4]{386., 937., 936., 2487.};
    expected.setData(data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test1();
    test2();
    test3();
}