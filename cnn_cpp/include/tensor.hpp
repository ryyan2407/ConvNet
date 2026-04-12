#pragma once

#include <memory>
#include <vector>

#include "shape.hpp"

class Tensor {
public:
    Tensor();
    Tensor(int n, int c, int h, int w);

    float& operator()(int n, int c, int h, int w);
    float operator()(int n, int c, int h, int w) const;

    int N() const;
    int C() const;
    int H() const;
    int W() const;

    Shape shape() const;
    int size() const;
    void fill(float value);
    void print_shape() const;
    void print_data() const;
    Tensor slice_n(int start_n, int count_n) const;

    const std::vector<float>& data() const;
    std::vector<float>& data();
    const float* raw_data() const;
    float* raw_data();
    int offset_unchecked(int n, int c, int h, int w) const;
    float& at_unchecked(int n, int c, int h, int w);
    float at_unchecked(int n, int c, int h, int w) const;

private:
    int n_;
    int c_;
    int h_;
    int w_;
    std::shared_ptr<std::vector<float>> storage_;
    std::size_t base_offset_;

    Tensor(int n, int c, int h, int w, std::shared_ptr<std::vector<float>> storage, std::size_t base_offset);

    int offset(int n, int c, int h, int w) const;
};
