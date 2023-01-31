#include "loader.h"
#include "batchnorm.h"
#include <iostream>

using namespace std;

void test_BN() {
    cout << "==== test BatchNorm ====" << endl;  

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

    // input.init(1, 2, 1, 2);
    // float input_data[4]{
    //     -0.3572,  0.2753, -0.4522, -0.7150};

    // input.init(1, 2, 2, 2);
    // float input_data[8]{
    //     1.0594, -0.2994,  0.9339, -0.9727,  2.1594,  0.1584,  0.7436,  0.3791};

    input.setData(input_data);

    BatchNorm2D bn;
    bn.loadGamma("../data/test_data.json", "weight");
    bn.loadBeta("../data/test_data.json", "bias");

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

    // expected.init(1, 2, 1, 2);
    // float expected_data[4]{
    //      2.0005, 5.9995, 7.9957, 2.0043};

    // expected.init(1, 2, 2, 2);
    // float expected_data[8]{
    //      6.0647,  2.8733,  5.7700,  1.2920, 10.0053,  2.2967,  4.5511,  3.1469};

    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;




    // Loader<float> weight;
    // weight.setFileName("../data/metr_epoch_33_2.8.json");
    // weight.setItemName("bn.0.weight");
    // weight.load();  

    // int weight_dim = weight.getDim();
    // cout << "weight dimension: " << weight_dim << ", shape: ";
    // for (int i = 0; i < weight_dim; i++) {
    //     cout << weight.getShape()[i] << " ";
    // }
    // cout << endl;

    // Loader<float> bias;
    // bias.setFileName("../data/metr_epoch_33_2.8.json");
    // bias.setItemName("bn.0.bias");
    // bias.load();  

    // int bias_dim = bias.getDim();
    // cout << "bias dimension: " << bias_dim << ", shape: ";
    // for (int i = 0; i < bias_dim; i++) {
    //     cout << bias.getShape()[i] << " ";
    // }
    // cout << endl;

    // Loader<float> running_mean;
    // running_mean.setFileName("../data/metr_epoch_33_2.8.json");
    // running_mean.setItemName("bn.0.running_mean");
    // running_mean.load();  

    // int mean_dim = running_mean.getDim();
    // cout << "running_mean dimension: " << mean_dim << ", shape: ";
    // for (int i = 0; i < mean_dim; i++) {
    //     cout << running_mean.getShape()[i] << " ";
    // }
    // cout << endl;

    // Loader<float> running_var;
    // running_var.setFileName("../data/metr_epoch_33_2.8.json");
    // running_var.setItemName("bn.0.running_var");
    // running_var.load();  

    // int var_dim = running_var.getDim();
    // cout << "running_var dimension: " << var_dim << ", shape: ";
    // for (int i = 0; i < var_dim; i++) {
    //     cout << running_var.getShape()[i] << " ";
    // }
    // cout << endl;

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

