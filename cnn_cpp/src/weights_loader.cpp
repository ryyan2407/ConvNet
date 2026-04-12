#include "weights_loader.hpp"

#include <fstream>
#include <stdexcept>

#include "utils.hpp"

std::vector<float> load_weights_from_file(const std::string& path, std::size_t expected_size) {
    auto values = read_floats_from_file(path);
    assert_expected_size(values.size(), expected_size, path);
    return values;
}

void save_weights_to_file(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        file << values.at(i);
        if (i + 1 < values.size()) {
            file << "\n";
        }
    }
}
