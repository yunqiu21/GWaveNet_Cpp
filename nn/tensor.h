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
    Tensor();
    ~Tensor();

    // copy constructor, use deep copy
    Tensor operator=(const Tensor &t);
    bool copyData(const Tensor &src);

    bool isSame(const Tensor &src);

    // initializaiton
    void init(const int *s, int d);
    void init(int W);                      // 1D Tensor
    void init(int H, int W);               // 2D Tensor
    void init(int C, int H, int W);        // 3D Tensor
    void init(int N, int C, int H, int W); // 4D Tensor

    bool isInit();

    T &operator()(int idx);                    // 1D Tensor
    T &operator()(int row, int col);           // 2D Tensor
    T &operator()(int C, int H, int W);        // 3D Tensor
    T &operator()(int N, int C, int H, int W); // 4D Tensor

    // data iterator
    T *begin();
    void next(T *&cur);

    // getters
    int getDim();
    const int *const getShape();

    void setData(T *d);
};

template <typename T>
Tensor<T>::Tensor() {
    data = nullptr;
    shape = nullptr;
    dim = 0;
};

template <typename T>
Tensor<T>::~Tensor() {
    if (data) {
        delete[] data;
        data = nullptr;
    }
    if (shape) {
        delete[] shape;
        shape = nullptr;
    }
};

// copy constructor, use deep copy
template <typename T>
Tensor<T> Tensor<T>::operator=(const Tensor<T> &t) {
    Tensor newTensor;
    newTensor.init(t.shape, t.dim);
    memcpy(newTensor.data, t.data, t.dataCount * sizeof(T));
};

template <typename T>
bool Tensor<T>::copyData(const Tensor<T> &src) {
    // check shape
    if (src.dim == dim && memcmp(src.shape, shape, dim * sizeof(T)) == 0) {
        memcpy(data, src.data, dataCount * sizeof(T));
        return true;
    }
    return false;
}

template <typename T>
bool Tensor<T>::isSame(const Tensor &src) {
    // check shape
    if (!(src.dim == dim &&
        memcmp(src.shape, shape, dim * sizeof(T)) == 0)) {
        return false;
    }

    for (int i = 0; i < dataCount; i++) {
        if (src.data[i] - data[i] >= 1e-4 || src.data[i] - data[i] <= -1e-4) {
            return false;
        }
    }

    return true;
}

template <typename T>
void Tensor<T>::init(const int *s, int d) {
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

template <typename T>
void Tensor<T>::init(int size) {
    dim = 1;
    shape = new int[1]{size};
    dataCount = size;
    data = new T[size];
};

template <typename T>
void Tensor<T>::init(int row, int column) {
    dim = 2;
    shape = new int[2]{row, column};
    dataCount = row * column;
    data = new T[dataCount];
};

template <typename T>
void Tensor<T>::init(int C, int H, int W) {
    dim = 3;
    shape = new int[3]{C, H, W};
    dataCount = C * H * W;
    data = new T[dataCount];
};

template <typename T>
void Tensor<T>::init(int N, int C, int H, int W) {
    dim = 4;
    shape = new int[4]{N, C, H, W};
    dataCount = N * C * H * W;
    data = new T[dataCount];
};

template <typename T>
bool Tensor<T>::isInit() {
    return data;
}

template <typename T>
int Tensor<T>::getDim() {
    return dim;
}

template <typename T>
const int *const Tensor<T>::getShape() {
    return shape;
}

template <typename T>
T &Tensor<T>::operator()(int idx) {
    assert(dim == 1 && idx < shape[0]);
    return *(data + idx);
};

template <typename T>
T &Tensor<T>::operator()(int row, int col) {
    assert(dim == 2 && row < shape[0] && col < shape[1]);
    return *(data + row * shape[1] + col);
};

template <typename T>
T &Tensor<T>::operator()(int C, int H, int W) {
    assert(dim == 3 && C < shape[0] && H < shape[1] && W < shape[2]);
    return *(data + C * shape[1] * shape[2] + H * shape[2] + W);
};

template <typename T>
T &Tensor<T>::operator()(int N, int C, int H, int W) {
    assert(dim == 4 && N < shape[0] && C < shape[1] && H < shape[2] && W < shape[3]);
    return *(data + N * shape[1] * shape[2] * shape[3] + C * shape[2] * shape[3] + H * shape[3] + W);
};

template <typename T>
T *Tensor<T>::begin() {
    return data;
}

template <typename T>
void Tensor<T>::next(T *&cur) {
    if (data && cur - data + 1 < dataCount) {
        cur++;
    } else {
        cur = nullptr;
    }
}

template <typename T>
void Tensor<T>::setData(T *d) {
    memcpy(data, d, dataCount * sizeof(T));
}
