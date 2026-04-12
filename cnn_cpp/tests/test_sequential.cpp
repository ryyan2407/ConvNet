#include <cassert>
#include <memory>
#include <stdexcept>

#include "flatten.hpp"
#include "relu.hpp"
#include "sequential.hpp"

int main() {
    Sequential empty_model;
    Tensor input(1, 1, 1, 2);
    bool threw = false;
    try {
        (void)empty_model.forward(input);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    Tensor values(1, 1, 1, 3);
    values(0, 0, 0, 0) = -2.0f;
    values(0, 0, 0, 1) = 1.0f;
    values(0, 0, 0, 2) = 3.0f;

    Sequential model;
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Flatten>());

    Tensor output = model.forward(values);
    assert(output.N() == 1);
    assert(output.C() == 1);
    assert(output.H() == 1);
    assert(output.W() == 3);
    assert(output(0, 0, 0, 0) == 0.0f);
    assert(output(0, 0, 0, 1) == 1.0f);
    assert(output(0, 0, 0, 2) == 3.0f);

    Tensor predicted = model.predict(values);
    assert(predicted(0, 0, 0, 0) == 0.0f);
    assert(predicted(0, 0, 0, 1) == 1.0f);
    assert(predicted(0, 0, 0, 2) == 3.0f);

    return 0;
}
