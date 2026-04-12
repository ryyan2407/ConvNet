#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

float random_float(float min_value, float max_value) {
    static std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(min_value, max_value);
    return distribution(generator);
}

bool nearly_equal(float a, float b, float epsilon) {
    return std::abs(a - b) <= epsilon;
}

std::vector<float> read_floats_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<float> values;
    float value = 0.0f;
    while (file >> value) {
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("No float values found in file: " + path);
    }
    return values;
}

void assert_expected_size(std::size_t actual, std::size_t expected, const std::string& label) {
    if (actual != expected) {
        throw std::runtime_error(label + " size mismatch. Expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

int compute_output_dim(int input_size, int kernel_size, int stride, int padding, const std::string& layer_name) {
    if (input_size <= 0) {
        throw std::runtime_error(layer_name + " input dimension must be positive");
    }

    const int numerator = input_size + 2 * padding - kernel_size;
    if (numerator < 0) {
        throw std::runtime_error(layer_name + " kernel is larger than the effective input");
    }
    if (numerator % stride != 0) {
        throw std::runtime_error(layer_name + " configuration does not tile the input exactly");
    }

    return numerator / stride + 1;
}
