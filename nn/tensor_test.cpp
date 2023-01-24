#include "tensor.h"
#include <iostream>

using namespace std;

int main() {
    Tensor<int> t;

    cout << t.isInit() << endl;
    cout << t.getShape() << endl;

    t.init(3, 2, 2);
    int num = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                t(i, j, k) = num;
                num++;
            }
        }
    }

    int *val = t.getFirst();
    while (val) {
        cout << "*" << *val << endl;
        t.getNext(val);
    }

    return 0;
}