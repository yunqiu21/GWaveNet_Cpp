// convolutional layer in C++
// official pytorch implementation:
//   https://github.com/pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py

#include "module.h"
#include <cmath>
#include <random>

class _ConvNd : public Module {
private:
    const int in_channels;  // input channel
    const int out_channels; // output channel
    const int kernel_h;     // not a general definition, kernal is 2 dimentional
    const int kernel_w;
    const int dilation;

    float *weight = nullptr; // shape (kernel_h * kernel_w)
    float *bias = nullptr;   // shape (out_channels)

    void reset_parameters() { // parameter initialization
        int n = in_channels * kernel_h * kernel_w;
        float stdv = 1 / sqrt(n);

        // random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-stdv, stdv);

        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                weight.at(i, j) = dis(gen);
            }
        }

        if (bias) {
            for (int i = 0; i < out_channels; i++) {
                bias[i] = dis(gen);
            }
        }
    }

public:
    _ConvNd(int in_channels, int out_channels, int kernel_h, int kernel_w, int dilation, bool bias) : in_channels(in_channels), out_channels(out_channels), m_kernelH(kernelH), m_kernelW(kernelW), m_dilation(dilation) {
    }

}

class Conv2d : public _ConvNd {
private:
public:
    Conv2d(int in_channels, int out_channels, int kernelH, int kernelW, int dilation = 1, bool bias = true)
        : m_in_channels(in_channels), m_out_channels(out_channels),
          m_kernelH(kernelH), m_kernelW(kernelW), m_dilation(dilation) {
        m_weight = new float[m_kernelH][m_kernelW];
        m_bias = new float[m_out_channels];
    }

    ~Conv2d() {
        delete[] m_weight;
        delete[] m_bias;
    }

    const int outChannels() const {
        return m_out_channels;
    }

    const int Dilation() const {
        return m_dilation;
    }

    const int KernelH() const {
        return m_kernelH;
    }

    const int KernelW() const {
        return m_kernelW;
    }

    void forward(float *input, int inH, int inW, int minibatch,
                 float *output, int outH, int outW) {
        /* convolution */
        // input: [minibatch][m_in_channels][inH][inW]
        // output: [minibatch][m_out_channels][inH-m_dilation(m_kernelH-1)][inW-m_dilation(m_kernelW-1)]
        for (int m = 0; m < minibatch; m++) {
            /* set bias */
            for (int i = 0; i < m_out_channels; i++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        output[m][i][h][w] = m_bias[i];
                    }
                }
            }
            /* convolution */
            for (int i = 0; i < m_out_channels; i++) {
                for (int j = 0; j < m_in_channels; j++) {
                    for (int h = 0; h < outH; h++) {
                        for (int w = 0; w < outW; w++) {
                            for (int p = 0; p < m_kernelH; p++) {
                                for (int q = 0; q < m_kernelW; q++) {
                                    output[m][i][h][w] += input[m][j][h + p * d][w + q * d] * m_weight[i][j][p][q];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void backward(){};
};