#include "gcn.h"
#include "nn/loader.h"
#include "list.h"

#include <iostream>

using namespace std;

void test_gcn() {
    cout << "==== test gcn ====" << endl;
    /* declare gcn class */
    GCN gcn(2,1);
    gcn.load("data/test_data.json", "gconv.mlp.mlp.weight", "gconv.mlp.mlp.bias");
    /* load x, support */
    Tensor<float> x;
    x.init(2,2,3,5);
    float x_data[60]{
        -0.0865,  0.0194, -0.7198, -1.3064,  0.4679, -0.5583, -0.5058, -1.2496,
        -0.1417,  1.7925, -0.5095, -0.6528,  0.9676,  0.2013, -1.4495, -0.1064,
        -1.7182, -1.0932,  1.2530,  0.2698, -1.4022, -0.5537,  2.0462,  0.2437,
         1.2503, -1.7572,  0.5026,  1.6914, -0.3388,  0.6981, -0.7136,  0.6665,
         0.6714,  0.7490,  0.4754,  0.0725, -1.0816,  1.3718, -0.6054,  1.2531,
         0.2663, -0.7511, -0.8578, -0.6528, -0.0890, -0.8227, -0.8683, -0.5724,
         0.1978,  1.0399, -2.0217,  1.9208,  0.7178, -1.7369, -0.0659, -0.3337,
        -1.2640, -0.9317, -0.7178, -1.0136};
    x.setData(x_data);
    List<Tensor<float>> support(3);
    Tensor<float> support0, support1, support2;
    support0.init(3,3);
    support1.init(3,3);
    support2.init(3,3);
    float support0_data[9]{
        -0.3280, -0.8905, -0.2023, -1.6437,  0.7402,
         0.8593, -0.3836,  0.8085,  1.0041};
    float support1_data[9]{
         0.1014,  0.0580,  0.4500,  0.6613, -0.4308,
         0.5420,  0.3102,  0.3018,  0.6507};
    float support2_data[9]{
         1.1244, -0.2427, -1.3910,  1.0413, -0.1463,
         0.4146, -1.7214, -0.5879, -1.9002};
    support0.setData(support0_data);
    support1.setData(support1_data);
    support2.setData(support2_data);
    support.add(support0);
    support.add(support1);
    support.add(support2);
    /* run gcn forward */
    Tensor<float> output;
    gcn.forward(x, support, output);
    /* check output dimension */
    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;
    /* compare with expected */
    Tensor<float> expected;
    expected.init(2,1,3,5);
    float expected_data[30]{
         1.2058420181e+01, -2.0124309540e+01, -3.6840179443e+01,
         1.7083444595e+00,  1.1664067268e+01, -3.0519134521e+01,
        -4.1218070984e+00,  3.2483612061e+01,  1.7741898298e+00,
         1.7694002151e+01, -4.5565986633e+01, -2.0874080658e+00,
         5.5327816010e+01, -4.6461343765e-02, -1.8341710567e+00,
        -1.6938867569e+01,  6.9294433594e+00,  8.4871959686e+00,
         2.4873716354e+01,  2.6906316757e+01, -1.0487666130e+01,
        -7.3240203857e+00,  3.3826580048e+00, -2.8346240997e+01,
        -5.4992353916e-01, -3.9641003609e+00, -3.1118162155e+01,
        -2.3046863556e+01, -3.2332431793e+01, -1.5814095497e+01};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test_gcn();
}