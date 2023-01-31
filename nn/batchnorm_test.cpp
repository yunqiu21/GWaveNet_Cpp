#include "loader.h"
#include <iostream>

using namespace std;

void test_BN() {
    cout << "==== test BatchNorm ====" << endl;  

    Loader<float> weight;
    weight.setFileName("../data/metr_epoch_33_2.8.json");
    weight.setItemName("bn.0.weight");
    weight.load();  

    int weight_dim = weight.getDim();
    cout << "weight dimension: " << weight_dim << ", shape: ";
    for (int i = 0; i < weight_dim; i++) {
        cout << weight.getShape()[i] << " ";
    }
    cout << endl;

    Loader<float> bias;
    bias.setFileName("../data/metr_epoch_33_2.8.json");
    bias.setItemName("bn.0.bias");
    bias.load();  

    int bias_dim = bias.getDim();
    cout << "bias dimension: " << bias_dim << ", shape: ";
    for (int i = 0; i < bias_dim; i++) {
        cout << bias.getShape()[i] << " ";
    }
    cout << endl;

    Loader<float> running_mean;
    running_mean.setFileName("../data/metr_epoch_33_2.8.json");
    running_mean.setItemName("bn.0.running_mean");
    running_mean.load();  

    int mean_dim = running_mean.getDim();
    cout << "running_mean dimension: " << mean_dim << ", shape: ";
    for (int i = 0; i < mean_dim; i++) {
        cout << running_mean.getShape()[i] << " ";
    }
    cout << endl;

    Loader<float> running_var;
    running_var.setFileName("../data/metr_epoch_33_2.8.json");
    running_var.setItemName("bn.0.running_var");
    running_var.load();  

    int var_dim = running_var.getDim();
    cout << "running_var dimension: " << var_dim << ", shape: ";
    for (int i = 0; i < var_dim; i++) {
        cout << running_var.getShape()[i] << " ";
    }
    cout << endl;

    // Tensor<float> expected;
    // expected.init(32);
    // float expected_data[32]{
    //     0.090507872402668, 0.09305255860090256, 0.21226881444454193,
    //     0.061750996857881546, -0.042060598731040955, -0.019422639161348343,
    //     -0.04646080359816551, -0.02996070683002472, -0.04182992875576019,
    //     0.009472918696701527, -0.22710733115673065, 0.05523635447025299,
    //     -0.15856371819972992, 0.0034840572625398636, 0.17996127903461456,
    //     0.027946777641773224, 0.07486802339553833, 0.07549183815717697,
    //     0.05077587440609932, 0.03902988135814667, -0.49055686593055725,
    //     -0.10872425138950348, 0.08546419441699982, 0.0076241809874773026,
    //     0.11949726939201355, 0.062061842530965805, -0.2085609883069992,
    //     -0.2672024965286255, 0.02411043830215931, -0.11112712323665619,
    //     -0.027019299566745758, -0.0050720395520329475};
    // expected.setData(expected_data);
    // cout << "same as expected: " << expected.isSame(loader) << endl;
}

int main() {
    test_BN();
}

