#include "model_config.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "conv2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "relu.hpp"
#include "softmax.hpp"

namespace {

std::unordered_map<std::string, std::string> parse_key_values(std::istringstream& stream) {
    std::unordered_map<std::string, std::string> values;
    std::string token;
    while (stream >> token) {
        const std::size_t pos = token.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid config token: " + token);
        }
        values[token.substr(0, pos)] = token.substr(pos + 1);
    }
    return values;
}

int require_int(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("Missing config key: " + key);
    }
    return std::stoi(it->second);
}

}  // namespace

Sequential build_model_from_config(const std::string& config_path) {
    std::ifstream config(config_path);
    if (!config) {
        throw std::runtime_error("Failed to open model config: " + config_path);
    }

    std::string header;
    std::getline(config, header);
    if (header != "cnn_cpp_config_v1") {
        throw std::runtime_error("Unsupported model config header: " + header);
    }

    Sequential model;
    std::string line;
    while (std::getline(config, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream line_stream(line);
        std::string record_type;
        std::string layer_type;
        line_stream >> record_type >> layer_type;
        if (record_type != "layer") {
            throw std::runtime_error("Unknown config record: " + line);
        }
        const auto values = parse_key_values(line_stream);

        if (layer_type == "Conv2D") {
            model.add(std::make_unique<Conv2D>(require_int(values, "in_channels"),
                                               require_int(values, "out_channels"),
                                               require_int(values, "kernel_size"),
                                               require_int(values, "stride"),
                                               require_int(values, "padding")));
        } else if (layer_type == "ReLU") {
            model.add(std::make_unique<ReLU>());
        } else if (layer_type == "MaxPool2D") {
            model.add(std::make_unique<MaxPool2D>(require_int(values, "kernel_size"),
                                                  require_int(values, "stride")));
        } else if (layer_type == "Flatten") {
            model.add(std::make_unique<Flatten>());
        } else if (layer_type == "Linear") {
            model.add(std::make_unique<Linear>(require_int(values, "in_features"),
                                               require_int(values, "out_features")));
        } else if (layer_type == "Softmax") {
            model.add(std::make_unique<Softmax>());
        } else {
            throw std::runtime_error("Unsupported layer type in config: " + layer_type);
        }
    }

    return model;
}
