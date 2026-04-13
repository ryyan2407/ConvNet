#include "sequential.hpp"

#include <stdexcept>

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }

    Tensor current = input;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

Tensor Sequential::predict(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }

    Tensor current = input;
    for (const auto& layer : layers_) {
        current = layer->infer(current);
    }
    return current;
}

Tensor Sequential::backward(const Tensor& grad_output) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }

    Tensor current = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        current = (*it)->backward(current);
    }
    return current;
}

void Sequential::zero_grad() {
    for (const auto& layer : layers_) {
        layer->zero_grad();
    }
}

void Sequential::update(float learning_rate) {
    for (const auto& layer : layers_) {
        layer->update(learning_rate);
    }
}

const std::vector<std::unique_ptr<Layer>>& Sequential::layers() const { return layers_; }
