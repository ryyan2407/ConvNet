#pragma once

#include <string>
#include <vector>

std::vector<float> load_weights_from_file(const std::string& path, std::size_t expected_size);
void save_weights_to_file(const std::string& path, const std::vector<float>& values);
