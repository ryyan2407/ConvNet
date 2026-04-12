#include "conv2d.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "utils.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kSpatialTile = 16;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int kernel_extent(int in_channels, int kernel_size) {
    return in_channels * kernel_size * kernel_size;
}

void im2col_sample(const float* input,
                   int in_channels,
                   int input_h,
                   int input_w,
                   int kernel_size,
                   int stride,
                   int padding,
                   int output_h,
                   int output_w,
                   std::vector<float>& columns) {
    const int output_spatial = output_h * output_w;
    const int extent = kernel_extent(in_channels, kernel_size);
    columns.assign(static_cast<std::size_t>(extent * output_spatial), 0.0f);

    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_channel_base = ic * input_h * input_w;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int row = (ic * kernel_size + kh) * kernel_size + kw;
                const int row_base = row * output_spatial;
                for (int oh = 0; oh < output_h; ++oh) {
                    const int ih = oh * stride + kh - padding;
                    const int col_row_base = row_base + oh * output_w;
                    if (ih < 0 || ih >= input_h) {
                        continue;
                    }

                    const int input_row_base = input_channel_base + ih * input_w;
                    if (stride == 1) {
                        const int ow_start = std::max(0, padding - kw);
                        const int ow_end = std::min(output_w, input_w + padding - kw);
                        const int length = ow_end - ow_start;
                        if (length > 0) {
                            const int iw_start = ow_start + kw - padding;
                            std::memcpy(columns.data() + static_cast<std::size_t>(col_row_base + ow_start),
                                        input + input_row_base + iw_start,
                                        static_cast<std::size_t>(length) * sizeof(float));
                        }
                    } else {
                        for (int ow = 0; ow < output_w; ++ow) {
                            const int iw = ow * stride + kw - padding;
                            if (iw >= 0 && iw < input_w) {
                                columns[static_cast<std::size_t>(col_row_base + ow)] =
                                    input[input_row_base + iw];
                            }
                        }
                    }
                }
            }
        }
    }
}

void col2im_sample(const std::vector<float>& columns,
                   int in_channels,
                   int input_h,
                   int input_w,
                   int kernel_size,
                   int stride,
                   int padding,
                   int output_h,
                   int output_w,
                   float* grad_input) {
    const int output_spatial = output_h * output_w;

    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_channel_base = ic * input_h * input_w;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int row = (ic * kernel_size + kh) * kernel_size + kw;
                const int row_base = row * output_spatial;
                for (int oh = 0; oh < output_h; ++oh) {
                    const int ih = oh * stride + kh - padding;
                    const int col_row_base = row_base + oh * output_w;
                    if (ih < 0 || ih >= input_h) {
                        continue;
                    }

                    const int input_row_base = input_channel_base + ih * input_w;
                    if (stride == 1) {
                        const int ow_start = std::max(0, padding - kw);
                        const int ow_end = std::min(output_w, input_w + padding - kw);
                        for (int ow = ow_start; ow < ow_end; ++ow) {
                            const int iw = ow + kw - padding;
                            grad_input[input_row_base + iw] +=
                                columns[static_cast<std::size_t>(col_row_base + ow)];
                        }
                    } else {
                        for (int ow = 0; ow < output_w; ++ow) {
                            const int iw = ow * stride + kw - padding;
                            if (iw >= 0 && iw < input_w) {
                                grad_input[input_row_base + iw] +=
                                    columns[static_cast<std::size_t>(col_row_base + ow)];
                            }
                        }
                    }
                }
            }
        }
    }
}

void gemm_forward(const std::vector<float>& weights,
                  const std::vector<float>& bias,
                  int out_channels,
                  int extent,
                  const std::vector<float>& columns,
                  int output_spatial,
                  float* output) {
    for (int oc = 0; oc < out_channels; ++oc) {
        const float* weight_row = weights.data() + static_cast<std::size_t>(oc * extent);
        float* output_row = output + oc * output_spatial;
        const float bias_value = bias[static_cast<std::size_t>(oc)];
        for (int p0 = 0; p0 < output_spatial; p0 += kSpatialTile) {
            const int pend = std::min(output_spatial, p0 + kSpatialTile);
            for (int p = p0; p < pend; ++p) {
                output_row[p] = bias_value;
            }
            for (int k = 0; k < extent; ++k) {
                const float weight = weight_row[k];
                const float* column_row = columns.data() + static_cast<std::size_t>(k * output_spatial) + p0;
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp simd
#endif
                for (int p = p0; p < pend; ++p) {
                    output_row[p] += weight * column_row[p - p0];
                }
            }
        }
    }
}

void transpose_columns(const std::vector<float>& columns,
                       int extent,
                       int output_spatial,
                       std::vector<float>& transposed) {
    transposed.resize(static_cast<std::size_t>(extent * output_spatial));
    for (int k = 0; k < extent; ++k) {
        const float* column_row = columns.data() + static_cast<std::size_t>(k * output_spatial);
        for (int p = 0; p < output_spatial; ++p) {
            transposed[static_cast<std::size_t>(p * extent + k)] = column_row[p];
        }
    }
}

void accumulate_weight_grads(const float* grad_output,
                             const std::vector<float>& columns_t,
                             int out_channels,
                             int extent,
                             int output_spatial,
                             std::vector<float>& grad_weights,
                             std::vector<float>& grad_bias) {
    for (int oc = 0; oc < out_channels; ++oc) {
        const float* grad_output_row = grad_output + oc * output_spatial;
        float bias_sum = 0.0f;
        float* grad_weight_row = grad_weights.data() + static_cast<std::size_t>(oc * extent);
        for (int p = 0; p < output_spatial; ++p) {
            const float grad = grad_output_row[p];
            bias_sum += grad;
            const float* packed_row = columns_t.data() + static_cast<std::size_t>(p * extent);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp simd
#endif
            for (int k = 0; k < extent; ++k) {
                grad_weight_row[k] += grad * packed_row[k];
            }
        }
        grad_bias[static_cast<std::size_t>(oc)] += bias_sum;
    }
}

void gemm_backward_input(const std::vector<float>& weights,
                         int out_channels,
                         int extent,
                         const float* grad_output,
                         int output_spatial,
                         std::vector<float>& grad_columns) {
    grad_columns.assign(static_cast<std::size_t>(extent * output_spatial), 0.0f);
    for (int oc = 0; oc < out_channels; ++oc) {
        const float* grad_output_row = grad_output + oc * output_spatial;
        const float* weight_row = weights.data() + static_cast<std::size_t>(oc * extent);
        for (int k = 0; k < extent; ++k) {
            const float weight = weight_row[k];
            float* grad_column_row = grad_columns.data() + static_cast<std::size_t>(k * output_spatial);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp simd
#endif
            for (int p = 0; p < output_spatial; ++p) {
                grad_column_row[p] += weight * grad_output_row[p];
            }
        }
    }
}

}  // namespace

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      weights_(expected_weight_count(), 0.0f),
      bias_(expected_bias_count(), 0.0f),
      grad_weights_(expected_weight_count(), 0.0f),
      grad_bias_(expected_bias_count(), 0.0f) {
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
        throw std::invalid_argument("Invalid Conv2D configuration");
    }

    for (float& weight : weights_) {
        weight = random_float(-0.1f, 0.1f);
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    cached_input_ = input;
    return run_forward(input);
}

Tensor Conv2D::infer(const Tensor& input) {
    return run_forward(input);
}

Tensor Conv2D::run_forward(const Tensor& input) const {
    if (input.C() != in_channels_) {
        throw std::runtime_error("Conv2D input channel mismatch");
    }

    const int output_h = compute_output_dim(input.H(), kernel_size_, stride_, padding_, "Conv2D");
    const int output_w = compute_output_dim(input.W(), kernel_size_, stride_, padding_, "Conv2D");
    const int output_spatial = output_h * output_w;
    const int extent = kernel_extent(in_channels_, kernel_size_);

    Tensor output(input.N(), out_channels_, output_h, output_w);
    const float* input_data = input.raw_data();
    float* output_data = output.raw_data();

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < input.N(); ++n) {
        std::vector<float> columns;
        const float* input_sample = input_data + input.offset_unchecked(n, 0, 0, 0);
        float* output_sample = output_data + output.offset_unchecked(n, 0, 0, 0);
        const auto im2col_start = Clock::now();
        im2col_sample(input_sample, in_channels_, input.H(), input.W(), kernel_size_, stride_, padding_,
                      output_h, output_w, columns);
        const auto im2col_end = Clock::now();
        const auto gemm_start = im2col_end;
        gemm_forward(weights_, bias_, out_channels_, extent, columns, output_spatial, output_sample);
        const auto gemm_end = Clock::now();

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
        profile_stats_.forward_im2col_ms += elapsed_ms(im2col_start, im2col_end);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
        profile_stats_.forward_gemm_ms += elapsed_ms(gemm_start, gemm_end);
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    zero_grad();
    Tensor grad_input(cached_input_.N(), cached_input_.C(), cached_input_.H(), cached_input_.W());
    grad_input.fill(0.0f);

    const int output_h = grad_output.H();
    const int output_w = grad_output.W();
    const int output_spatial = output_h * output_w;
    const int extent = kernel_extent(in_channels_, kernel_size_);
    const float* cached_input_data = cached_input_.raw_data();
    const float* grad_output_data = grad_output.raw_data();
    float* grad_input_data = grad_input.raw_data();

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<float> local_grad_weights(grad_weights_.size(), 0.0f);
        std::vector<float> local_grad_bias(grad_bias_.size(), 0.0f);
        std::vector<float> columns;
        std::vector<float> columns_t;
        std::vector<float> grad_columns;

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int n = 0; n < cached_input_.N(); ++n) {
            const float* input_sample = cached_input_data + cached_input_.offset_unchecked(n, 0, 0, 0);
            const float* grad_output_sample = grad_output_data + grad_output.offset_unchecked(n, 0, 0, 0);
            float* grad_input_sample = grad_input_data + grad_input.offset_unchecked(n, 0, 0, 0);

            const auto im2col_start = Clock::now();
            im2col_sample(input_sample, in_channels_, cached_input_.H(), cached_input_.W(), kernel_size_, stride_,
                          padding_, output_h, output_w, columns);
            const auto im2col_end = Clock::now();
            const auto grad_accum_start = im2col_end;
            transpose_columns(columns, extent, output_spatial, columns_t);
            accumulate_weight_grads(grad_output_sample, columns_t, out_channels_, extent, output_spatial,
                                    local_grad_weights, local_grad_bias);
            const auto grad_accum_end = Clock::now();
            const auto gemm_start = grad_accum_end;
            gemm_backward_input(weights_, out_channels_, extent, grad_output_sample, output_spatial, grad_columns);
            const auto gemm_end = Clock::now();
            const auto col2im_start = gemm_end;
            col2im_sample(grad_columns, in_channels_, cached_input_.H(), cached_input_.W(), kernel_size_, stride_,
                          padding_, output_h, output_w, grad_input_sample);
            const auto col2im_end = Clock::now();

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
            profile_stats_.backward_im2col_ms += elapsed_ms(im2col_start, im2col_end);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
            profile_stats_.backward_grad_accum_ms += elapsed_ms(grad_accum_start, grad_accum_end);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
            profile_stats_.backward_gemm_ms += elapsed_ms(gemm_start, gemm_end);
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp atomic
#endif
            profile_stats_.backward_col2im_ms += elapsed_ms(col2im_start, col2im_end);
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
        }
    }

    return grad_input;
}

void Conv2D::zero_grad() {
    std::fill(grad_weights_.begin(), grad_weights_.end(), 0.0f);
    std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

void Conv2D::update(float learning_rate) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_weights_[i];
    }
    for (std::size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_bias_[i];
    }
}

void Conv2D::set_weights(const std::vector<float>& weights) {
    assert_expected_size(weights.size(), static_cast<std::size_t>(expected_weight_count()), "Conv2D weights");
    weights_ = weights;
}

void Conv2D::set_bias(const std::vector<float>& bias) {
    assert_expected_size(bias.size(), static_cast<std::size_t>(expected_bias_count()), "Conv2D bias");
    bias_ = bias;
}

const std::vector<float>& Conv2D::weights() const { return weights_; }

const std::vector<float>& Conv2D::bias() const { return bias_; }

const Conv2D::ProfileStats& Conv2D::profile_stats() const { return profile_stats_; }

void Conv2D::reset_profile_stats() { profile_stats_ = {}; }

int Conv2D::expected_weight_count() const {
    return out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
}

int Conv2D::expected_bias_count() const { return out_channels_; }

float Conv2D::weight_at(int oc, int ic, int kh, int kw) const {
    const int index = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;
    return weights_[static_cast<std::size_t>(index)];
}
