#include "gwavenet.h"
#include "list.h"
#include "nn/activation.h"
#include "nn/loader.h"
#include "nn/matmul.h"
#include <chrono>
#include <iostream>

using namespace std;

void test_gwavenet() {
    cout << "==== test gwavenet ====" << endl;
    /* declare gwavenet class */
    GWaveNet gwavenet(207);
    /* load gwavenet parameters */
    cout << "begin loading" << endl;
    chrono::steady_clock::time_point load_begin = chrono::steady_clock::now();
    gwavenet.load("data/metr_exp1_best_2.73.json");
    chrono::steady_clock::time_point load_end = chrono::steady_clock::now();
    cout << "Load parameters finished in " << (chrono::duration_cast<chrono::nanoseconds>(load_end - load_begin).count()) * 1e-9 << " seconds." << endl;
    /* load input */
    Loader<float> loader;
    Tensor<float> input;
    loader.setFileName("data/input.json");
    loader.setItemName("testx");
    loader.load(input);  
    /* check input dimension */
    int in_dim = input.getDim();
    cout << "input dimension: " << in_dim << ", shape: ";
    for (int i = 0; i < in_dim; i++) {
        cout << input.getShape()[i] << " ";
    }
    cout << endl;
    /* run gwavenet forward */
    Tensor<float> output;
    chrono::steady_clock::time_point forward_begin = chrono::steady_clock::now();
    gwavenet.forward(input, output);
    chrono::steady_clock::time_point forward_end = chrono::steady_clock::now();
    cout << "Inference finished in " << (chrono::duration_cast<chrono::nanoseconds> (forward_end - forward_begin).count()) * 1e-9 << " seconds." << endl;
    /* check output dimension */
    int dim = output.getDim();
    cout << "output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;
    /* compare with expected */
    Tensor<float> expected;
    loader.setFileName("data/expected.json");
    loader.setItemName("preds");
    loader.load(expected);
    cout << "error: " << expected.MSE(output) << endl;
}

int main() {
    test_gwavenet();
}