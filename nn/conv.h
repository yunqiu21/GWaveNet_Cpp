// convolutional layer in C++
// official pytorch implementation:
//   https://github.com/pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py

#include "assert.h"
#include "module.h"
#include <cmath>
#include <iostream>
#include <random>
#include <string.h>

using namespace std;

class _ConvNd : public Module {
protected:
    const int in_channels;  // input channel
    const int out_channels; // output channel
    const int kernel_h;     // not a general definition, kernal is 2 dimentional
    const int kernel_w;
    const int dilation;

    Tensor<float> weight; // shape (kernel_h * kernel_w)
    Tensor<float> bias;   // shape (out_channels)

    void reset_parameters() { // parameter initialization
        int n = in_channels * kernel_h * kernel_w;
        float stdv = 1 / sqrt(n);

        // random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-stdv, stdv);

        {
            float *w = weight.getFirst();
            while (w) {
                *w = dis(gen);
                w = weight.getNext(w);
            }
        }

        if (bias.isInit()) {
            float *b = bias.getFirst();
            while (b) {
                *b = dis(gen);
                b = bias.getNext(b);
            }
        }
    }

public:
    _ConvNd(int in_channels, int out_channels, int kernel_h, int kernel_w,
            int dilation, bool needBias) : in_channels(in_channels), out_channels(out_channels),
                                           kernel_h(kernel_h), kernel_w(kernel_w), dilation(dilation) {
        if (kernel_w == 0) {
            weight.init(out_channels, in_channels, kernel_h); // for conv1D
        } else {
            weight.init(out_channels, in_channels, kernel_h, kernel_w); // for conv2D
        }

        if (needBias) {
            bias.init(out_channels);
        }
        reset_parameters();
    };

    const int getOutChannels() const {
        return out_channels;
    }

    const int getDilation() const {
        return dilation;
    }

    const int getKernelH() const {
        assert(kernel_h != 0);
        return kernel_h;
    }

    const int getKernelW() const {
        assert(kernel_w != 0);
        return kernel_w;
    }

    void setWeight(Tensor<float> &w) {
        weight.copyData(w);
    }

    void setBias(Tensor<float> &b) {
        bias.copyData(b);
    }

    virtual void forward(Tensor<float> &input, Tensor<float> &output) = 0;
    virtual void backward() = 0;
};

class Conv1d : public _ConvNd {
public:
    Conv1d(int in_channels, int out_channels, int kernel,
           int dilation = 1, bool bias = true) : _ConvNd(in_channels, out_channels, 0, kernel, dilation, bias) {}

    // input: [N][in_channels][L]
    // output: [N][out_channels][L-m_dilation(kernelW-1)]
    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.isInit() && input.getDim() == 3);
        assert(!output.isInit());
        const int *shape = input.getShape();
        int N = shape[0];
        int C = shape[1];
        int L = shape[2];

        /* calulate output shape */
        int outL = L - dilation * (kernel_w - 1);
        output.init(N, C, outL);

        for (int n = 0; n < N; n++) {
            /* set bias */
            for (int c = 0; c < out_channels; c++) {
                for (int h = 0; h < outL; h++) {
                    for (int w = 0; w < outL; w++) {
                        output(n, c, h, w) = bias(c); // change to memset
                    }
                }
            }

            /* convolution */
            for (int i = 0; i < out_channels; i++) {
                for (int j = 0; j < in_channels; j++) {
                    for (int l = 0; l < outL; l++) {
                        for (int q = 0; q < kernel_w; q++) {
                            output(n, i, l) += input(n, j, l + q * dilation) * weight(i, j, q);
                        }
                    }
                }
            }
        }
    }

    void backward() {}
};

class Conv2d : public _ConvNd {
public:
    Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int dilation = 1, bool bias = true) : _ConvNd(in_channels, out_channels, kernel_h, kernel_w, dilation, bias) {}

    // input: [N][in_channels][H][W]
    // output: [N][out_channels][H-dilation(KernalH-1)][W-dilation(kernelW-1)]
    void forward(Tensor<float> &input, Tensor<float> &output) {
        assert(input.isInit() && input.getDim() == 4);
        assert(!output.isInit());
        const int *shape = input.getShape();
        int N = shape[0];
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];
        cout << N << C << H << W << endl;

        /* calulate output shape */
        int outH = H - dilation * (kernel_h - 1);
        int outW = W - dilation * (kernel_w - 1);
        output.init(N, out_channels, outH, outW);

        for (int n = 0; n < N; n++) {
            /* set bias */
            for (int c = 0; c < out_channels; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        output(n, c, h, w) = bias(c); // change to memset
                    }
                }
            }

            /* convolution */
            for (int i = 0; i < out_channels; i++) {
                for (int j = 0; j < in_channels; j++) {
                    for (int h = 0; h < outH; h++) {
                        for (int w = 0; w < outW; w++) {
                            for (int p = 0; p < kernel_h; p++) {
                                for (int q = 0; q < kernel_w; q++) {
                                    output(n, i, h, w) += input(n, j, h + p * dilation, w + q * dilation) * weight(i, j, p, q);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void backward() {}
};