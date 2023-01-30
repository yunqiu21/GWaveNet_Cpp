#include "module.h"
#include <cassert>

// element wise multiplication
void mul(Tensor<float> &t1, Tensor<float> &t2, Tensor<float> &output) {
    output = t1;
    float *it1 = output.begin();
    float *it2 = t2.begin();
    while (it1 && it2) {
        *it1 *= *it2;
        output.next(it1);
        t2.next(it2);
    }
}

// element wise multiplication 2D
void matmul2D(Tensor<float> &t1, Tensor<float> &t2, Tensor<float> &output) {
    assert(t1.getDim() == 2 && t2.getDim() == 2 && t1.getShape()[1] == t2.getShape()[0]);
    const int M = t1.getShape()[0];
    const int K = t1.getShape()[1];
    const int N = t2.getShape()[1];

    output.init(M, N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            output(m, n) = 0;
            for (int k = 0; k < K; k++) {
                output(m, n) += t1(m, k) * t2(k, n);
            }
        }
    }
}