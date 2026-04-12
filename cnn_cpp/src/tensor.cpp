#include "tensor.hpp"

#include <iostream>
#include <stdexcept>

Tensor::Tensor() : n_(0), c_(0), h_(0), w_(0), storage_(std::make_shared<std::vector<float>>()), base_offset_(0) {}

Tensor::Tensor(int n, int c, int h, int w)
    : n_(n),
      c_(c),
      h_(h),
      w_(w),
      storage_(std::make_shared<std::vector<float>>(static_cast<std::size_t>(n * c * h * w), 0.0f)),
      base_offset_(0) {
    if (n < 0 || c < 0 || h < 0 || w < 0) {
        throw std::invalid_argument("Tensor dimensions must be non-negative");
    }
}

Tensor::Tensor(int n, int c, int h, int w, std::shared_ptr<std::vector<float>> storage, std::size_t base_offset)
    : n_(n), c_(c), h_(h), w_(w), storage_(std::move(storage)), base_offset_(base_offset) {}

float& Tensor::operator()(int n, int c, int h, int w) {
    return raw_data()[offset(n, c, h, w)];
}

float Tensor::operator()(int n, int c, int h, int w) const {
    return raw_data()[offset(n, c, h, w)];
}

int Tensor::N() const { return n_; }
int Tensor::C() const { return c_; }
int Tensor::H() const { return h_; }
int Tensor::W() const { return w_; }

Shape Tensor::shape() const { return Shape{n_, c_, h_, w_}; }

int Tensor::size() const { return n_ * c_ * h_ * w_; }

void Tensor::fill(float value) {
    std::fill(raw_data(), raw_data() + size(), value);
}

void Tensor::print_shape() const {
    std::cout << "[" << n_ << ", " << c_ << ", " << h_ << ", " << w_ << "]\n";
}

void Tensor::print_data() const {
    for (int n = 0; n < n_; ++n) {
        for (int c = 0; c < c_; ++c) {
            std::cout << "Tensor(n=" << n << ", c=" << c << ")\n";
            for (int h = 0; h < h_; ++h) {
                for (int w = 0; w < w_; ++w) {
                    std::cout << (*this)(n, c, h, w) << " ";
                }
                std::cout << "\n";
            }
        }
    }
}

Tensor Tensor::slice_n(int start_n, int count_n) const {
    if (start_n < 0 || count_n < 0 || start_n + count_n > n_) {
        throw std::out_of_range("Tensor slice out of range");
    }
    const std::size_t sample_size = static_cast<std::size_t>(c_ * h_ * w_);
    return Tensor(count_n, c_, h_, w_, storage_, base_offset_ + static_cast<std::size_t>(start_n) * sample_size);
}

const std::vector<float>& Tensor::data() const {
    if (base_offset_ != 0 || storage_->size() != static_cast<std::size_t>(size())) {
        throw std::runtime_error("data() is only available for owning contiguous tensors");
    }
    return *storage_;
}

std::vector<float>& Tensor::data() {
    if (base_offset_ != 0 || storage_->size() != static_cast<std::size_t>(size())) {
        throw std::runtime_error("data() is only available for owning contiguous tensors");
    }
    return *storage_;
}

const float* Tensor::raw_data() const { return storage_->data() + base_offset_; }
float* Tensor::raw_data() { return storage_->data() + base_offset_; }

int Tensor::offset_unchecked(int n, int c, int h, int w) const {
    return ((n * c_ + c) * h_ + h) * w_ + w;
}

float& Tensor::at_unchecked(int n, int c, int h, int w) {
    return raw_data()[offset_unchecked(n, c, h, w)];
}

float Tensor::at_unchecked(int n, int c, int h, int w) const {
    return raw_data()[offset_unchecked(n, c, h, w)];
}

int Tensor::offset(int n, int c, int h, int w) const {
    if (n < 0 || n >= n_ || c < 0 || c >= c_ || h < 0 || h >= h_ || w < 0 || w >= w_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return offset_unchecked(n, c, h, w);
}
