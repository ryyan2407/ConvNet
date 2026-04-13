#include "linear.hpp"

#include <stdexcept>

#include "utils.hpp"

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(expected_weight_count(), 0.0f),
      bias_(expected_bias_count(), 0.0f),
      grad_weights_(expected_weight_count(), 0.0f),
      grad_bias_(expected_bias_count(), 0.0f) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Invalid Linear configuration");
    }

    for (float& weight : weights_) {
        weight = random_float(-0.1f, 0.1f);
    }
}

Tensor Linear::forward(const Tensor& input) {
    cached_input_ = input;
    return run_forward(input);
}

Tensor Linear::infer(const Tensor& input) {
    return run_forward(input);
}

Tensor Linear::run_forward(const Tensor& input) const {
    if (input.C() != 1 || input.H() != 1 || input.W() != in_features_) {
        throw std::runtime_error("Linear expects input shape [N, 1, 1, in_features]");
    }

    Tensor output(input.N(), 1, 1, out_features_);
    const float* input_data = input.raw_data();
    float* output_data = output.raw_data();
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int n = 0; n < input.N(); ++n) {
        const int input_base = input.offset_unchecked(n, 0, 0, 0);
        const int output_base = output.offset_unchecked(n, 0, 0, 0);
        for (int of = 0; of < out_features_; ++of) {
            float sum = bias_[static_cast<std::size_t>(of)];
            const int weight_base = of * in_features_;
            for (int inf = 0; inf < in_features_; ++inf) {
                sum += input_data[input_base + inf] * weights_[static_cast<std::size_t>(weight_base + inf)];
            }
            output_data[output_base + of] = sum;
        }
    }
    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    zero_grad();
    Tensor grad_input(cached_input_.N(), 1, 1, in_features_);
    grad_input.fill(0.0f);
    const float* cached_input_data = cached_input_.raw_data();
    const float* grad_output_data = grad_output.raw_data();
    float* grad_input_data = grad_input.raw_data();

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<float> local_grad_weights(grad_weights_.size(), 0.0f);
        std::vector<float> local_grad_bias(grad_bias_.size(), 0.0f);
        std::vector<float> local_grad_input(static_cast<std::size_t>(grad_input.size()), 0.0f);

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int n = 0; n < cached_input_.N(); ++n) {
            const int input_base = cached_input_.offset_unchecked(n, 0, 0, 0);
            const int grad_output_base = grad_output.offset_unchecked(n, 0, 0, 0);
            const int grad_input_base = grad_input.offset_unchecked(n, 0, 0, 0);
            for (int of = 0; of < out_features_; ++of) {
                const float grad = grad_output_data[grad_output_base + of];
                local_grad_bias[static_cast<std::size_t>(of)] += grad;
                const int weight_base = of * in_features_;
                for (int inf = 0; inf < in_features_; ++inf) {
                    local_grad_weights[static_cast<std::size_t>(weight_base + inf)] +=
                        cached_input_data[input_base + inf] * grad;
                    local_grad_input[static_cast<std::size_t>(grad_input_base + inf)] +=
                        weights_[static_cast<std::size_t>(weight_base + inf)] * grad;
                }
            }
        }

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp critical
#endif
        {
            for (std::size_t i = 0; i < grad_weights_.size(); ++i) {
                grad_weights_[i] += local_grad_weights[i];
            }
            for (std::size_t i = 0; i < grad_bias_.size(); ++i) {
                grad_bias_[i] += local_grad_bias[i];
            }
            for (std::size_t i = 0; i < static_cast<std::size_t>(grad_input.size()); ++i) {
                grad_input_data[i] += local_grad_input[i];
            }
        }
    }

    return grad_input;
}

void Linear::zero_grad() {
    std::fill(grad_weights_.begin(), grad_weights_.end(), 0.0f);
    std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

void Linear::update(float learning_rate) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_weights_[i];
    }
    for (std::size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_bias_[i];
    }
}

void Linear::set_weights(const std::vector<float>& weights) {
    assert_expected_size(weights.size(), static_cast<std::size_t>(expected_weight_count()), "Linear weights");
    weights_ = weights;
}

void Linear::set_bias(const std::vector<float>& bias) {
    assert_expected_size(bias.size(), static_cast<std::size_t>(expected_bias_count()), "Linear bias");
    bias_ = bias;
}

const std::vector<float>& Linear::weights() const { return weights_; }

const std::vector<float>& Linear::bias() const { return bias_; }

const std::vector<float>& Linear::grad_weights() const { return grad_weights_; }

const std::vector<float>& Linear::grad_bias() const { return grad_bias_; }

int Linear::in_features() const { return in_features_; }

int Linear::out_features() const { return out_features_; }

int Linear::expected_weight_count() const { return out_features_ * in_features_; }
int Linear::expected_bias_count() const { return out_features_; }

float Linear::weight_at(int out_feature, int in_feature) const {
    return weights_[static_cast<std::size_t>(out_feature * in_features_ + in_feature)];
}
