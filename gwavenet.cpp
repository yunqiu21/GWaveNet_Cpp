#include "adp.h"
#include "gcn.h"
#include "nn/batchnorm.h"
#include <cassert>

class GWaveNet {
private:
    const int dropout;
    const int blocks;
    const int layers;
    const bool gcn_bool;
    const bool addaptadj;

    Conv2d start_conv;

    List<Conv2d> filter_convs;
    List<Tanh> filter_activations;

    List<Conv2d> gate_convs;
    List<Sigmoid> gate_activations;

    List<Conv2d> residual_convs;
    List<Conv2d> skip_convs;
    List<BatchNorm2D> bn;
    List<GCN> gconv;

    Conv2d end_conv1;
    ReLU end_relu1;
    Conv2d end_conv2;
    ReLU end_relu2;

    List<Tensor<float>> supports;

    Adp adp;
    bool haveAdp;

    int receptive_field;

public:
    GWaveNet(int num_nodes, float dropout = 0.3,
             List<Tensor<float>> supports = List<Tensor<float>>(),
             bool gcn_bool = true, bool addaptadj = true,
             int in_dim = 2, int out_dim = 12,
             int residual_channels = 32, int dilation_channels = 32,
             int skip_channels = 256, int end_channels = 512,
             int kernel_size = 2, int blocks = 4, int layers = 2)
        : dropout(dropout), supports(supports), blocks(blocks), layers(layers), gcn_bool(gcn_bool), addaptadj(addaptadj),
          start_conv(Conv2d(in_dim, residual_channels, 1, 1)),
          adp(Adp(num_nodes)),
          end_conv1(Conv2d(skip_channels, end_channels, 1, 1)),
          end_conv2(Conv2d(end_channels, out_dim, 1, 1)) {
        int receptive_field = 1;

        if (gcn_bool && addaptadj) {
            // suppose aptinit is none
            adp.randomInit();
            haveAdp = true;
        }

        for (int b = 0; b < blocks; b++) {
            int additional_scope = kernel_size - 1;
            int new_dilation = 1;
            for (int l = 0; l < layers; l++) {
                filter_convs.add(Conv2d(residual_channels, dilation_channels, 1, kernel_size, new_dilation));
                filter_activations.add(Tanh());
                gate_convs.add(Conv2d(residual_channels, dilation_channels, 1, kernel_size, new_dilation));
                gate_activations.add(Sigmoid());

                residual_convs.add(Conv2d(dilation_channels, residual_channels, 1, 1));
                skip_convs.add(Conv2d(dilation_channels, skip_channels, 1, 1));

                bn.add(BatchNorm2D(residual_channels));

                new_dilation *= 2;
                receptive_field += additional_scope;
                additional_scope *= 2;

                if (gcn_bool)
                    gconv.add(GCN(dilation_channels, residual_channels, dropout, supports.size() + int(haveAdp)));
            }
        }
    };

    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.getDim() == 4 && input.getShape()[3] > receptive_field);

        Tensor<float> x;
        start_conv.forward(input, x);

        Tensor<float> skip;

        List<Tensor<float>> new_supports = supports;
        if (gcn_bool && addaptadj && supports.size() > 0) {
            Tensor<float> adp_out;
            adp.forward(adp_out);
            new_supports.add(adp_out);
        }

        for (int i = 0; i < blocks * layers; i++) {
            Tensor<float> residual = x;

            Tensor<float> filter_temp;
            filter_convs(i).forward(residual, filter_temp);
            Tensor<float> filter;
            filter_activations(i).forward(filter_temp, filter);

            Tensor<float> gate_temp;
            gate_convs(i).forward(residual, gate_temp);
            Tensor<float> gate;
            gate_activations(i).forward(gate_temp, gate);

            mul(filter, gate, x);

            Tensor<float> s;
            skip_convs(i).forward(x, s);

            if (skip.reshapeLastDim(s.getShape()[3])) {
                skip = skip + s;
            } else {
                skip = s;
            }

            Tensor<float> x1;
            if (gcn_bool && supports.size() > 0) {
                gconv(i).forward(x, supports, x1);
            } else {
                residual_convs(i).forward(x, x1);
            }

            residual.reshapeLastDim(x.getShape()[3]);
            x1 = x1 + residual;

            bn(i).forward(x1, x);
        }

        Tensor<float> skip_relu;
        end_relu1.forward(skip, skip_relu);
        Tensor<float> end1;
        end_conv1.forward(skip_relu, end1);
        Tensor<float> skip_relu2;
        end_relu2.forward(end1, skip_relu2);
        Tensor<float> end2;
        end_conv1.forward(skip_relu2, output);
    };
};

int main() {
    GWaveNet(207);
}