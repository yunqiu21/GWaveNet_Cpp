#include "adp.h"
#include "batchnorm.h"
#include "gcn.h"
#include "list.h"
#include "module.h"
#include "nn/conv.h"

class GWaveNet {
private:
    const int dropout;
    const int blocks;
    const int layers;
    const bool gcn_bool;
    const bool addaptadj;

    Conv2d start_conv;

    List<Conv2d> filter_convs;
    List<Conv2d> gate_convs;
    List<Conv2d> residual_convs;
    List<Conv2d> skip_convs;
    List<BatchNorm2D> bn;
    List<GCN> gconv;

    Conv2d end_conv1;
    Conv2d end_conv2;

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
                gate_convs.add(Conv2d(residual_channels, dilation_channels, 1, kernel_size, new_dilation));

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

    void forward(Tensor<float> &input, Tensor<float> &output){};
};