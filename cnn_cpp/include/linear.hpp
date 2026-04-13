#pragma once

#include <vector>

#include "layer.hpp"

class Linear : public Layer {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input) override;
    Tensor infer(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void zero_grad() override;
    void update(float learning_rate) override;

    void set_weights(const std::vector<float>& weights);
    void set_bias(const std::vector<float>& bias);
    const std::vector<float>& weights() const;
    const std::vector<float>& bias() const;
    const std::vector<float>& grad_weights() const;
    const std::vector<float>& grad_bias() const;

    int in_features() const;
    int out_features() const;
    int expected_weight_count() const;
    int expected_bias_count() const;

private:
    int in_features_;
    int out_features_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> grad_weights_;
    std::vector<float> grad_bias_;
    Tensor cached_input_;

    float weight_at(int out_feature, int in_feature) const;
    Tensor run_forward(const Tensor& input) const;
};
