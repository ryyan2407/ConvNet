#include "sgd.hpp"

SGD::SGD(float learning_rate) : learning_rate_(learning_rate) {}

void SGD::zero_grad(Sequential& model) {
    model.zero_grad();
}

void SGD::step(Sequential& model) {
    model.update(learning_rate_);
}
