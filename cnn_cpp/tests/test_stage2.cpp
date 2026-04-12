#include <cassert>
#include <memory>
#include <vector>

#include "conv2d.hpp"
#include "cross_entropy.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "sgd.hpp"

namespace {

Tensor make_sample(const std::vector<float>& pixels) {
    Tensor sample(1, 1, 4, 4);
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            sample(0, 0, h, w) = pixels.at(static_cast<std::size_t>(h * 4 + w));
        }
    }
    return sample;
}

}  // namespace

int main() {
    Sequential model;
    model.add(std::make_unique<Conv2D>(1, 2, 3, 1, 1));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2));
    model.add(std::make_unique<Flatten>());
    model.add(std::make_unique<Linear>(8, 2));

    const Tensor sample = make_sample({
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1
    });

    CrossEntropyLoss loss;
    SGD optimizer(0.1f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    const int target = 0;
    for (int step = 0; step < 60; ++step) {
        optimizer.zero_grad(model);
        Tensor logits = model.forward(sample);
        const float current_loss = loss.forward(logits, LabelView{&target, 1});
        if (step == 0) {
            first_loss = current_loss;
        }
        last_loss = current_loss;
        Tensor gradient = loss.backward();
        Tensor grad_input = model.backward(gradient);
        assert(grad_input.N() == 1);
        assert(grad_input.C() == 1);
        assert(grad_input.H() == 4);
        assert(grad_input.W() == 4);
        optimizer.step(model);
    }

    assert(last_loss < first_loss);
    return 0;
}
