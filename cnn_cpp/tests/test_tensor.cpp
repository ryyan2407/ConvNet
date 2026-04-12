#include <cassert>

#include "tensor.hpp"

int main() {
    Tensor tensor(1, 2, 2, 3);
    assert(tensor.N() == 1);
    assert(tensor.C() == 2);
    assert(tensor.H() == 2);
    assert(tensor.W() == 3);
    assert(tensor.size() == 12);

    int value = 0;
    for (int c = 0; c < tensor.C(); ++c) {
        for (int h = 0; h < tensor.H(); ++h) {
            for (int w = 0; w < tensor.W(); ++w) {
                tensor(0, c, h, w) = static_cast<float>(value++);
            }
        }
    }

    assert(tensor(0, 0, 0, 0) == 0.0f);
    assert(tensor(0, 0, 1, 2) == 5.0f);
    assert(tensor(0, 1, 0, 0) == 6.0f);
    assert(tensor(0, 1, 1, 2) == 11.0f);

    tensor.fill(3.5f);
    for (int i = 0; i < tensor.size(); ++i) {
        assert(tensor.data().at(static_cast<std::size_t>(i)) == 3.5f);
    }

    return 0;
}
