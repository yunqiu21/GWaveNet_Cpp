#include "adp.h"
#include "nn/loader.h"
#include "nn/matmul.h"
#include "nn/activation.h"
#include <iostream>

using namespace std;

void test_adp() {
    cout << "==== test adp ====" << endl;
    /* declare adp class */
    Adp adp(6);
    /* load nodevec1, nodevec2 */
    adp.load("data/test_data.json", "nodevec1", "nodevec2");
    /* run adp forward */
    Tensor<float> output;
    adp.forward(output);
    /* check output dimension */
    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;
    /* compare with expected */
    Tensor<float> expected;
    expected.init(6, 6);
    float expected_data[36]{
        0.0029, 0.0454, 0.0042, 0.0311, 0.0029, 0.9136, 0.2705, 0.0053, 0.2836,
        0.0055, 0.4128, 0.0224, 0.0806, 0.0806, 0.5969, 0.0806, 0.0806, 0.0806,
        0.0327, 0.3577, 0.0327, 0.0374, 0.0327, 0.5069, 0.0252, 0.0252, 0.8103,
        0.0466, 0.0676, 0.0252, 0.0575, 0.1117, 0.5206, 0.0575, 0.1337, 0.1189};
    expected.setData(expected_data);
    cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test_adp();
}