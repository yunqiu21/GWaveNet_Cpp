#include "matmul.h"
#include <iostream>

using namespace std;

void test_matmul() {
    cout << "==== test matrix multiplication ====" << endl;

    Tensor<float> t1;
    t1.init(4, 5);
    float t1_data[20]{
         0.0674, -1.0838,  0.2145, -1.2188, -1.2034,  2.4634,  0.7295, -0.8012,
        -0.0117,  1.2978, -0.3526, -1.5425, -0.7073,  1.3479, -1.0193,  0.2188,
         2.8235, -1.2821, -0.9532, -0.0842};
    t1.setData(t1_data);

    Tensor<float> t2;
    t2.init(5, 4);
    float t2_data[20]{
        -1.8476,  0.2254, -0.2072,  0.8800,  0.9048,  0.4829, -0.5440, -0.3767,
        -1.0221,  1.0486,  0.5709, -0.5094,  0.1110, -0.0487, -0.0022,  0.6921,
         0.0144,  1.0423,  0.5960, -1.1460};
    t2.setData(t2_data);

    Tensor<float> output;
    matmul2D(t1, t2, output);

    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;

    Tensor<float> expected;
    expected.init(4, 4);
    float expected_data[16]{
        -1.4770, -1.4782, -0.0165,  0.8939, -3.0550,  1.4207, -0.5912,  0.8057,
         0.1137, -2.6941, -0.1021,  2.7321,  3.3539,  0.0270, -2.3614, -0.7812};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test_matmul();
}