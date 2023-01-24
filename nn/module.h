#include "tensor.h"

// TODO: Use RealType?
class Module {
private:
public:
    virtual void forward(Tensor<float> &input, Tensor<float> &output){};
    virtual void backward(){};
};