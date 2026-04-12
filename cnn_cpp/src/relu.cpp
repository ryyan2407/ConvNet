#include "relu.hpp"

#include <algorithm>

namespace {

Tensor relu_forward_impl(const Tensor& input) {
    Tensor output(input.N(), input.C(), input.H(), input.W());
    const float* input_data = input.raw_data();
    float* output_data = output.raw_data();
    const int total = output.size();
    for (int i = 0; i < total; ++i) {
        output_data[i] = std::max(0.0f, input_data[i]);
    }
    return output;
}

}  // namespace

Tensor ReLU::forward(const Tensor& input) {
    cached_input_ = input;
    return relu_forward_impl(input);
}

Tensor ReLU::infer(const Tensor& input) { return relu_forward_impl(input); }

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input(cached_input_.N(), cached_input_.C(), cached_input_.H(), cached_input_.W());
    const float* cached_input_data = cached_input_.raw_data();
    const float* grad_output_data = grad_output.raw_data();
    float* grad_input_data = grad_input.raw_data();
    const int total = grad_input.size();
    for (int i = 0; i < total; ++i) {
        grad_input_data[i] = cached_input_data[i] > 0.0f ? grad_output_data[i] : 0.0f;
    }
    return grad_input;
}
