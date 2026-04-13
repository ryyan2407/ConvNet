#include "maxpool2d.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "utils.hpp"

MaxPool2D::MaxPool2D(int kernel_size, int stride) : kernel_size_(kernel_size), stride_(stride) {
    if (kernel_size <= 0 || stride <= 0) {
        throw std::invalid_argument("Invalid MaxPool2D configuration");
    }
}

Tensor MaxPool2D::forward(const Tensor& input) {
    cached_input_ = input;
    return run_forward(input, true);
}

Tensor MaxPool2D::infer(const Tensor& input) {
    return run_forward(input, false);
}

Tensor MaxPool2D::run_forward(const Tensor& input, bool store_indices) {
    const int output_h = compute_output_dim(input.H(), kernel_size_, stride_, 0, "MaxPool2D");
    const int output_w = compute_output_dim(input.W(), kernel_size_, stride_, 0, "MaxPool2D");
    const bool fast_2x2 = (kernel_size_ == 2 && stride_ == 2);

    Tensor output(input.N(), input.C(), output_h, output_w);
    const float* input_data = input.raw_data();
    float* output_data = output.raw_data();
    if (store_indices) {
        max_indices_.assign(static_cast<std::size_t>(output.size()), -1);
    }
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (int n = 0; n < input.N(); ++n) {
        for (int c = 0; c < input.C(); ++c) {
            const int input_channel_base = input.offset_unchecked(n, c, 0, 0);
            for (int oh = 0; oh < output_h; ++oh) {
                const int ih_origin = oh * stride_;
                for (int ow = 0; ow < output_w; ++ow) {
                    const int output_index = output.offset_unchecked(n, c, oh, ow);
                    if (fast_2x2) {
                        const int iw_origin = ow << 1;
                        const int row0 = input_channel_base + ih_origin * input.W() + iw_origin;
                        const int row1 = row0 + input.W();
                        const float v0 = input_data[row0];
                        const float v1 = input_data[row0 + 1];
                        const float v2 = input_data[row1];
                        const float v3 = input_data[row1 + 1];

                        float max_value = v0;
                        int max_index = row0;
                        if (v1 > max_value) {
                            max_value = v1;
                            max_index = row0 + 1;
                        }
                        if (v2 > max_value) {
                            max_value = v2;
                            max_index = row1;
                        }
                        if (v3 > max_value) {
                            max_value = v3;
                            max_index = row1 + 1;
                        }

                        output_data[output_index] = max_value;
                        if (store_indices) {
                            max_indices_[static_cast<std::size_t>(output_index)] = max_index;
                        }
                    } else {
                        const int iw_origin = ow * stride_;
                        float max_value = std::numeric_limits<float>::lowest();
                        int max_index = -1;
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            const int input_row_base = input_channel_base + (ih_origin + kh) * input.W();
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int input_index = input_row_base + iw_origin + kw;
                                const float candidate = input_data[input_index];
                                if (candidate > max_value) {
                                    max_value = candidate;
                                    max_index = input_index;
                                }
                            }
                        }

                        output_data[output_index] = max_value;
                        if (store_indices) {
                            max_indices_[static_cast<std::size_t>(output_index)] = max_index;
                        }
                    }
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_input(cached_input_.N(), cached_input_.C(), cached_input_.H(), cached_input_.W());
    grad_input.fill(0.0f);
    const float* grad_output_data = grad_output.raw_data();
    float* grad_input_data = grad_input.raw_data();
    const bool non_overlapping = stride_ >= kernel_size_;

    const int output_h = grad_output.H();
    const int output_w = grad_output.W();
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel for collapse(2) if(non_overlapping)
#endif
    for (int n = 0; n < grad_output.N(); ++n) {
        for (int c = 0; c < grad_output.C(); ++c) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    const int output_index = grad_output.offset_unchecked(n, c, oh, ow);
                    const int input_index = max_indices_[static_cast<std::size_t>(output_index)];
                    if (non_overlapping) {
                        grad_input_data[input_index] = grad_output_data[output_index];
                    } else {
                        grad_input_data[input_index] += grad_output_data[output_index];
                    }
                }
            }
        }
    }

    return grad_input;
}

int MaxPool2D::kernel_size() const { return kernel_size_; }

int MaxPool2D::stride() const { return stride_; }
