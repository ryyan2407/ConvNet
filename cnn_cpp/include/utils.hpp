#pragma once

#include <cstddef>
#include <string>
#include <vector>

float random_float(float min_value, float max_value);
bool nearly_equal(float a, float b, float epsilon = 1e-5f);
std::vector<float> read_floats_from_file(const std::string& path);
void assert_expected_size(std::size_t actual, std::size_t expected, const std::string& label);
int compute_output_dim(int input_size, int kernel_size, int stride, int padding, const std::string& layer_name);
