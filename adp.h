#include "activation.h"
#include "matmul.h"
#include "module.h"

#define NUM_NODES 207

class Adp : public Module {
private:
    Tensor<float> nodevec1;
    Tensor<float> nodevec2;
    ReLU relu;
    Softmax softmax;

public:
    Adp() {
        nodevec1.init(NUM_NODES, 10);
        nodevec2.init(10, NUM_NODES);
    }

    void forward(Tensor<float> &output) {
        Tensor<float> out1;
        matmul2D(nodevec1, nodevec2, out1);

        Tensor<float> out2;
        relu.forward(out1, out2);

        softmax.forward(out2, output);
    }
};