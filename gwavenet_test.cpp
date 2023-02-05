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
    /* load supports */
    Loader<float> loader;
    List<Tensor<float>> supports;
    Tensor<float> support0;
    loader.setFileName("data/supports.json");
    loader.setItemName("supports.0");
    loader.load(support0);
    Tensor<float> support1;
    loader.setItemName("supports.1");
    loader.load(support1);
    supports.add(support0);
    supports.add(support1);
    /* declare gwavenet class */
    GWaveNet gwavenet(207, 0.3, supports);
    /* load gwavenet parameters */
    cout << "Begin load..." << endl;
    chrono::steady_clock::time_point load_begin = chrono::steady_clock::now();
    gwavenet.load("data/metr_exp1_best_2.73.json");
    chrono::steady_clock::time_point load_end = chrono::steady_clock::now();
    cout << "Load finished in " << (chrono::duration_cast<chrono::nanoseconds>(load_end - load_begin).count()) * 1e-9 << " seconds." << endl;
    /* load input */
    Tensor<float> input;
    loader.setFileName("data/input.json");
    loader.setItemName("testx");
    loader.load(input);
    /* check input dimension */
    int in_dim = input.getDim();
    cout << "Input dimension: " << in_dim << ", shape: ";
    for (int i = 0; i < in_dim; i++) {
        cout << input.getShape()[i] << " ";
    }
    cout << endl;
    /* run gwavenet forward */
    Tensor<float> output;
    cout << "Begin inference..." << endl;
    chrono::steady_clock::time_point forward_begin = chrono::steady_clock::now();
    gwavenet.forward(input, output);
    chrono::steady_clock::time_point forward_end = chrono::steady_clock::now();
    cout << "Inference finished in " << (chrono::duration_cast<chrono::nanoseconds> (forward_end - forward_begin).count()) * 1e-9 << " seconds." << endl;
    /* check output dimension */
    int dim = output.getDim();
    cout << "Output dimension: " << dim << ", shape: ";
    for (int i = 0; i < dim; i++) {
        cout << output.getShape()[i] << " ";
    }
    cout << endl;
    /* compare with expected */
    Tensor<float> expected;
    loader.setFileName("data/expected.json");
    loader.setItemName("preds");
    loader.load(expected);
    cout << "Same as expected: " << boolalpha << expected.isSame(output) << endl;
}

int main() {
    test_gwavenet();
}