#pragma once

#include <vector>

#include "layer.hpp"

class Conv2D : public Layer {
public:
    struct ProfileStats {
        double forward_im2col_ms = 0.0;
        double forward_gemm_ms = 0.0;
        double backward_im2col_ms = 0.0;
        double backward_grad_accum_ms = 0.0;
        double backward_gemm_ms = 0.0;
        double backward_col2im_ms = 0.0;
    };

    Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

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
    const ProfileStats& profile_stats() const;
    void reset_profile_stats();

    int in_channels() const;
    int out_channels() const;
    int kernel_size() const;
    int stride() const;
    int padding() const;
    int expected_weight_count() const;
    int expected_bias_count() const;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> grad_weights_;
    std::vector<float> grad_bias_;
    Tensor cached_input_;
    mutable ProfileStats profile_stats_;

    float weight_at(int oc, int ic, int kh, int kw) const;
    Tensor run_forward(const Tensor& input) const;
};
