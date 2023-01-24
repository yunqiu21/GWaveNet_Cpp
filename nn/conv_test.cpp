#include "conv.h"
#include <iostream>

using namespace std;
int main() {
    Tensor<float> input;
    input.init(2, 3, 4, 5);

    /* init input */
    int count = 1;
    float *i = input.getFirst();
    while (i) {
        *i = count;
        i = input.getNext(i);
        count++;
    }

    Conv2d conv = Conv2d(3, 2, 2, 2, 2, true);

    cout << "set weight" << endl;
    Tensor<float> weight;
    weight.init(2, 3, 2, 2);
    count = 1;
    i = weight.getFirst();
    while (i) {
        *i = count;
        i = weight.getNext(i);
        count++;
    }
    conv.setWeight(weight);

    Tensor<float> bias;
    bias.init(2);
    count = 1;
    i = bias.getFirst();
    while (i) {
        *i = count;
        i = bias.getNext(i);
        count++;
    }
    conv.setBias(bias);

    Tensor<float> output;
    cout << "start forward" << endl;
    conv.forward(input, output);

    int dim = output.getDim();
    cout << dim << endl;
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(2, 2, 2, 3);
}