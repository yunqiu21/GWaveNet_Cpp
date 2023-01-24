#include "nconv.h"
#include "nn/activation.h"
#include "nn/conv.h"

class GCN : public Module {
private:
    NConv nconv;
    const int c_in;
    Conv2d mlp;
    const int order;
    Dropout dropout;

    void concat(Tensor<float> &x, Tensor<float> &out, int &cIdx) {
        for (int n = 0; n < x.getShape()[0]; n++) {
            for (int c = 0; c < x.getShape()[1]; c++) {
                for (int h = 0; h < x.getShape()[2]; h++) {
                    for (int w = 0; w < x.getShape()[3]; w++) {
                        out(n, cIdx + c, h, w) = x(n, c, h, w);
                    }
                }
            }
        }
        cIdx += x.getShape()[1];
    };

public:
    GCN(int c_in, int c_out, float dropout = 0.3, int support_len = 3, int order = 2)
        : c_in((order * support_len + 1) * c_in), order(order), mlp(Conv2d(c_in, c_out, 1, 1)){};

    void forward(Tensor<float> &x, Tensor<float> *support, int supportSz, Tensor<float> &output) {
        Tensor<float> out;
        out.init(x.getShape()[0], x.getShape()[1] * (1 + supportSz * order), x.getShape()[2], x.getShape()[3]);

        int cIdx = 0;
        concat(x, out, cIdx);
        for (int i = 0; i < supportSz; i++) {
            Tensor<float> x1;
            nconv.forward(x, support[i], x1);
            concat(x1, out, cIdx);
            for (int j = 2; j < order + 1; j++) {
                Tensor<float> x2;
                nconv.forward(x1, support[i], x2);
                concat(x2, out, cIdx);
                x1 = x2;
            }
        }

        Tensor<float> out1;
        mlp.forward(out, out1);
        dropout.forward(out1, output);
    }
};