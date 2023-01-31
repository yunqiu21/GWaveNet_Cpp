#ifndef NCONV_H
#define NCONV_H

#include "nn/tensor.h"

class NConv {
public:
    // TODO: use sparse matrix multiplication
    void forward(Tensor<float> &x, Tensor<float> &A, Tensor<float> &output) {
        assert(x.getDim() == 4 && A.getDim() == 2);

        output.init(x.getShape()[0], x.getShape()[1], A.getShape()[1], x.getShape()[3]);
        for (int n = 0; n < x.getShape()[0]; n++) {
            for (int c = 0; c < x.getShape()[1]; c++) {
                for (int w = 0; w < A.getShape()[1]; w++) {
                    for (int l = 0; l < x.getShape()[3]; l++) {
                        float sum = 0;
                        for (int v = 0; v < x.getShape()[2]; v++) {
                            sum += x(n, c, v, l) * A(v, w);
                        }
                        output(n, c, w, l) = sum;
                    }
                }
            }
        }
    };
};

#endif