#include "assert.h"
#include <string.h>

// Tensor
template <typename T>
class Tensor {
protected:
    T *data;
    int dataCount;
    const int *shape;
    int dim;

public:
    Tensor() {
        data = nullptr;
        shape = nullptr;
        dim = 0;
    };

    // copy constructor, use deep copy
    Tensor operator=(const Tensor &t) {
        Tensor newTensor;
        newTensor.init(t.shape, t.dim);
        memcpy(newTensor.data, t.data, t.dataCount * sizeof(T));
    };

    bool copyData(const Tensor &src) {
        // check shape
        if (src.dim == dim && memcmp(src.shape, shape, dim * sizeof(T)) == 0) {
            memcpy(data, src.data, dataCount * sizeof(T));
            return true;
        }
        return false;
    }

    bool isSame(const Tensor &src) {
        // check shape
        return src.dim == dim &&
               memcmp(src.shape, shape, dim * sizeof(T)) == 0 &&
               memcmp(data, src.data, dataCount * sizeof(T)) == 0;
    }

    void init(const int *s, int d) {
        if (d == 0) {
            return;
        }
        dim = d;

        shape = new int[d];
        memcpy(shape, s, d * sizeof(int));

        dataCount = 1;
        for (int i = 0; i < dim; i++) {
            dataCount *= s[i];
        }

        data = new T[dataCount];
    };

    // 1D Tensor
    void init(int size) {
        dim = 1;
        shape = new int[1]{size};
        dataCount = size;
        data = new T[size];
    };

    // 2D Tensor
    void init(int row, int column) {
        dim = 2;
        shape = new int[2]{row, column};
        dataCount = row * column;
        data = new T[dataCount];
    };

    // 3D Tensor
    void init(int C, int H, int W) {
        dim = 3;
        shape = new int[3]{C, H, W};
        dataCount = C * H * W;
        data = new T[dataCount];
    };

    // 4D Tensor
    void init(int N, int C, int H, int W) {
        dim = 4;
        shape = new int[4]{N, C, H, W};
        dataCount = N * C * H * W;
        data = new T[dataCount];
    };

    bool isInit() {
        return data;
    }

    int getDim() {
        return dim;
    }

    const int *const getShape() {
        return shape;
    }

    // 1D Tensor
    T &operator()(int idx) {
        assert(dim == 1 && idx < shape[0]);
        return *(data + idx);
    };

    // 2D Tensor
    T &operator()(int row, int col) {
        assert(dim == 2 && row < shape[0] && col < shape[1]);
        return *(data + row * shape[1] + col);
    };

    // 3D Tensor
    T &operator()(int C, int H, int W) {
        assert(dim == 3 && C < shape[0] && H < shape[1] && W < shape[2]);
        return *(data + C * shape[1] * shape[2] + H * shape[2] + W);
    };

    // 4D Tensor
    T &operator()(int N, int C, int H, int W) {
        assert(dim == 4 && N < shape[0] && C < shape[1] && H < shape[2] && W < shape[3]);
        return *(data + N * shape[1] * shape[2] * shape[3] + C * shape[2] * shape[3] + H * shape[3] + W);
    };

    T *getFirst() {
        return data;
    }

    void getNext(T *&cur) {
        if (data && cur - data + 1 < dataCount) {
            cur++;
        }
        cur = nullptr;
    }

    ~Tensor() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        if (shape) {
            delete[] shape;
            shape = nullptr;
        }
    };
};
