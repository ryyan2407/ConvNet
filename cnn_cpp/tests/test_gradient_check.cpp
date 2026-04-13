#include <cassert>
#include <cmath>
#include <vector>

#include "conv2d.hpp"
#include "linear.hpp"

namespace {

constexpr float kEpsilon = 1e-3f;
constexpr float kTolerance = 2e-2f;

float scalar_objective(const Tensor& output, const Tensor& upstream) {
    assert(output.size() == upstream.size());
    float sum = 0.0f;
    for (int n = 0; n < output.N(); ++n) {
        for (int c = 0; c < output.C(); ++c) {
            for (int h = 0; h < output.H(); ++h) {
                for (int w = 0; w < output.W(); ++w) {
                    sum += output(n, c, h, w) * upstream(n, c, h, w);
                }
            }
        }
    }
    return sum;
}

void assert_close(float actual, float expected, float tolerance = kTolerance) {
    const float diff = std::fabs(actual - expected);
    assert(diff <= tolerance);
}

float numerical_linear_weight_grad(Linear& layer, const Tensor& input, const Tensor& upstream, int index) {
    std::vector<float> weights = layer.weights();
    weights[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_weights(weights);
    const float plus = scalar_objective(layer.infer(input), upstream);

    weights[static_cast<std::size_t>(index)] -= 2.0f * kEpsilon;
    layer.set_weights(weights);
    const float minus = scalar_objective(layer.infer(input), upstream);

    weights[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_weights(weights);
    return (plus - minus) / (2.0f * kEpsilon);
}

float numerical_linear_bias_grad(Linear& layer, const Tensor& input, const Tensor& upstream, int index) {
    std::vector<float> bias = layer.bias();
    bias[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_bias(bias);
    const float plus = scalar_objective(layer.infer(input), upstream);

    bias[static_cast<std::size_t>(index)] -= 2.0f * kEpsilon;
    layer.set_bias(bias);
    const float minus = scalar_objective(layer.infer(input), upstream);

    bias[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_bias(bias);
    return (plus - minus) / (2.0f * kEpsilon);
}

float numerical_linear_input_grad(const Linear& template_layer, Tensor input, const Tensor& upstream, int index) {
    input.raw_data()[index] += kEpsilon;
    Linear plus_layer = template_layer;
    const float plus = scalar_objective(plus_layer.infer(input), upstream);

    input.raw_data()[index] -= 2.0f * kEpsilon;
    Linear minus_layer = template_layer;
    const float minus = scalar_objective(minus_layer.infer(input), upstream);
    return (plus - minus) / (2.0f * kEpsilon);
}

float numerical_conv_weight_grad(Conv2D& layer, const Tensor& input, const Tensor& upstream, int index) {
    std::vector<float> weights = layer.weights();
    weights[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_weights(weights);
    const float plus = scalar_objective(layer.infer(input), upstream);

    weights[static_cast<std::size_t>(index)] -= 2.0f * kEpsilon;
    layer.set_weights(weights);
    const float minus = scalar_objective(layer.infer(input), upstream);

    weights[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_weights(weights);
    return (plus - minus) / (2.0f * kEpsilon);
}

float numerical_conv_bias_grad(Conv2D& layer, const Tensor& input, const Tensor& upstream, int index) {
    std::vector<float> bias = layer.bias();
    bias[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_bias(bias);
    const float plus = scalar_objective(layer.infer(input), upstream);

    bias[static_cast<std::size_t>(index)] -= 2.0f * kEpsilon;
    layer.set_bias(bias);
    const float minus = scalar_objective(layer.infer(input), upstream);

    bias[static_cast<std::size_t>(index)] += kEpsilon;
    layer.set_bias(bias);
    return (plus - minus) / (2.0f * kEpsilon);
}

float numerical_conv_input_grad(const Conv2D& template_layer, Tensor input, const Tensor& upstream, int index) {
    input.raw_data()[index] += kEpsilon;
    Conv2D plus_layer = template_layer;
    const float plus = scalar_objective(plus_layer.infer(input), upstream);

    input.raw_data()[index] -= 2.0f * kEpsilon;
    Conv2D minus_layer = template_layer;
    const float minus = scalar_objective(minus_layer.infer(input), upstream);
    return (plus - minus) / (2.0f * kEpsilon);
}

void test_linear_gradient_check() {
    Linear layer(3, 2);
    layer.set_weights({0.15f, -0.20f, 0.05f,
                       0.30f, 0.10f, -0.25f});
    layer.set_bias({0.01f, -0.02f});

    Tensor input(1, 1, 1, 3);
    input(0, 0, 0, 0) = 0.4f;
    input(0, 0, 0, 1) = -0.7f;
    input(0, 0, 0, 2) = 0.2f;

    Tensor upstream(1, 1, 1, 2);
    upstream(0, 0, 0, 0) = 0.6f;
    upstream(0, 0, 0, 1) = -0.4f;

    layer.zero_grad();
    layer.forward(input);
    Tensor analytic_input_grad = layer.backward(upstream);

    for (int i = 0; i < layer.expected_weight_count(); ++i) {
        const float numerical = numerical_linear_weight_grad(layer, input, upstream, i);
        assert_close(layer.grad_weights()[static_cast<std::size_t>(i)], numerical);
    }
    for (int i = 0; i < layer.expected_bias_count(); ++i) {
        const float numerical = numerical_linear_bias_grad(layer, input, upstream, i);
        assert_close(layer.grad_bias()[static_cast<std::size_t>(i)], numerical);
    }
    for (int i = 0; i < input.size(); ++i) {
        const float numerical = numerical_linear_input_grad(layer, input, upstream, i);
        assert_close(analytic_input_grad.raw_data()[i], numerical);
    }
}

void test_conv2d_gradient_check() {
    Conv2D layer(1, 1, 2, 1, 0);
    layer.set_weights({0.20f, -0.10f,
                       0.05f, 0.30f});
    layer.set_bias({0.07f});

    Tensor input(1, 1, 3, 3);
    const std::vector<float> values = {
        0.2f, -0.1f, 0.4f,
        0.0f, 0.3f, -0.2f,
        0.5f, -0.4f, 0.1f
    };
    for (int i = 0; i < input.size(); ++i) {
        input.raw_data()[i] = values[static_cast<std::size_t>(i)];
    }

    Tensor upstream(1, 1, 2, 2);
    upstream(0, 0, 0, 0) = 0.8f;
    upstream(0, 0, 0, 1) = -0.3f;
    upstream(0, 0, 1, 0) = 0.5f;
    upstream(0, 0, 1, 1) = -0.6f;

    layer.zero_grad();
    layer.forward(input);
    Tensor analytic_input_grad = layer.backward(upstream);

    for (int i = 0; i < layer.expected_weight_count(); ++i) {
        const float numerical = numerical_conv_weight_grad(layer, input, upstream, i);
        assert_close(layer.grad_weights()[static_cast<std::size_t>(i)], numerical);
    }

    for (int i = 0; i < layer.expected_bias_count(); ++i) {
        const float numerical = numerical_conv_bias_grad(layer, input, upstream, i);
        assert_close(layer.grad_bias()[static_cast<std::size_t>(i)], numerical);
    }

    for (int i = 0; i < input.size(); ++i) {
        const float numerical = numerical_conv_input_grad(layer, input, upstream, i);
        assert_close(analytic_input_grad.raw_data()[i], numerical);
    }
}

}  // namespace

int main() {
    test_linear_gradient_check();
    test_conv2d_gradient_check();
    return 0;
}
