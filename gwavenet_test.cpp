#include "gwavenet.h"
#include "nn/loader.h"
#include "nn/matmul.h"
#include "nn/activation.h"
#include "list.h"
#include <iostream>
#include <chrono>

using namespace std;

void test_gwavenet() {    
    cout << "==== test gwavenet ====" << endl;  
    /* set timer */
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();  
    /* declare gwavenet class */
    GWaveNet gwavenet(207);
    /* load nodevec1, nodevec2 */
    gwavenet.load("data/metr_exp1_best_2.73.json");
    chrono::steady_clock::time_point loaded = chrono::steady_clock::now();
    cout << "Loading finished in " << (chrono::duration_cast<chrono::nanoseconds> (loaded - begin).count()) * 1e-9 << " seconds." << endl;
    // /* run gwavenet forward */
    // Tensor<float> output;
    // gwavenet.forward(output);
    // chrono::steady_clock::time_point end = chrono::steady_clock::now();
    // cout << "Inference finished in " << (chrono::duration_cast<chrono::nanoseconds> (end - loaded).count()) * 1e-9 << " seconds." << endl;
    // /* check output dimension */
    // int dim = output.getDim();
    // cout << "output dimension: " << dim << ", shape: ";
    // for (int i = 0; i < dim; i++) {
    //     cout << output.getShape()[i] << " ";
    // }
    // cout << endl;
    // /* compare with expected */
    // Tensor<float> expected;
    // expected.init(6, 6);
    // float expected_data[36]{
    //     0.0029, 0.0454, 0.0042, 0.0311, 0.0029, 0.9136, 0.2705, 0.0053, 0.2836,
    //     0.0055, 0.4128, 0.0224, 0.0806, 0.0806, 0.5969, 0.0806, 0.0806, 0.0806,
    //     0.0327, 0.3577, 0.0327, 0.0374, 0.0327, 0.5069, 0.0252, 0.0252, 0.8103,
    //     0.0466, 0.0676, 0.0252, 0.0575, 0.1117, 0.5206, 0.0575, 0.1337, 0.1189};
    // expected.setData(expected_data);
    // cout << "same as expected: " << expected.isSame(output) << endl;
}

int main() {
    test_gwavenet();
}