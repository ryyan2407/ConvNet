#include "model_io.hpp"

#include <filesystem>
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
#include "weights_loader.hpp"

namespace {

std::unordered_map<std::string, std::string> parse_key_values(std::istringstream& stream) {
    std::unordered_map<std::string, std::string> values;
    std::string token;
    while (stream >> token) {
        const std::size_t pos = token.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid manifest token: " + token);
        }
        values[token.substr(0, pos)] = token.substr(pos + 1);
    }
    return values;
}

int require_int(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("Missing manifest key: " + key);
    }
    return std::stoi(it->second);
}

std::string require_string(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("Missing manifest key: " + key);
    }
    return it->second;
}

}  // namespace

void save_model_artifact(const Sequential& model,
                         const std::string& directory,
                         const std::string& manifest_filename) {
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root);
    std::ofstream manifest(root / manifest_filename);
    if (!manifest) {
        throw std::runtime_error("Failed to open manifest for writing: " + (root / manifest_filename).string());
    }

    manifest << "cnn_cpp_model_v1\n";
    int trainable_index = 0;
    for (const auto& layer_ptr : model.layers()) {
        if (const auto* conv = dynamic_cast<const Conv2D*>(layer_ptr.get())) {
            const std::string weights_name = "layer_" + std::to_string(trainable_index) + "_weights.txt";
            const std::string bias_name = "layer_" + std::to_string(trainable_index) + "_bias.txt";
            save_weights_to_file((root / weights_name).string(), conv->weights());
            save_weights_to_file((root / bias_name).string(), conv->bias());
            manifest << "layer Conv2D"
                     << " in_channels=" << conv->in_channels()
                     << " out_channels=" << conv->out_channels()
                     << " kernel_size=" << conv->kernel_size()
                     << " stride=" << conv->stride()
                     << " padding=" << conv->padding()
                     << " weights=" << weights_name
                     << " bias=" << bias_name << "\n";
            ++trainable_index;
        } else if (dynamic_cast<const ReLU*>(layer_ptr.get()) != nullptr) {
            manifest << "layer ReLU\n";
        } else if (const auto* pool = dynamic_cast<const MaxPool2D*>(layer_ptr.get())) {
            manifest << "layer MaxPool2D"
                     << " kernel_size=" << pool->kernel_size()
                     << " stride=" << pool->stride() << "\n";
        } else if (dynamic_cast<const Flatten*>(layer_ptr.get()) != nullptr) {
            manifest << "layer Flatten\n";
        } else if (const auto* linear = dynamic_cast<const Linear*>(layer_ptr.get())) {
            const std::string weights_name = "layer_" + std::to_string(trainable_index) + "_weights.txt";
            const std::string bias_name = "layer_" + std::to_string(trainable_index) + "_bias.txt";
            save_weights_to_file((root / weights_name).string(), linear->weights());
            save_weights_to_file((root / bias_name).string(), linear->bias());
            manifest << "layer Linear"
                     << " in_features=" << linear->in_features()
                     << " out_features=" << linear->out_features()
                     << " weights=" << weights_name
                     << " bias=" << bias_name << "\n";
            ++trainable_index;
        } else if (dynamic_cast<const Softmax*>(layer_ptr.get()) != nullptr) {
            manifest << "layer Softmax\n";
        } else {
            throw std::runtime_error("Unsupported layer type in save_model_artifact");
        }
    }
}

Sequential load_model_artifact(const std::string& manifest_path) {
    std::ifstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("Failed to open manifest for reading: " + manifest_path);
    }

    std::filesystem::path base_dir = std::filesystem::path(manifest_path).parent_path();
    std::string header;
    std::getline(manifest, header);
    if (header != "cnn_cpp_model_v1") {
        throw std::runtime_error("Unsupported model manifest header: " + header);
    }

    Sequential model;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream line_stream(line);
        std::string record_type;
        std::string layer_type;
        line_stream >> record_type >> layer_type;
        if (record_type != "layer") {
            throw std::runtime_error("Unknown manifest record: " + line);
        }
        const auto values = parse_key_values(line_stream);

        if (layer_type == "Conv2D") {
            auto layer = std::make_unique<Conv2D>(require_int(values, "in_channels"),
                                                  require_int(values, "out_channels"),
                                                  require_int(values, "kernel_size"),
                                                  require_int(values, "stride"),
                                                  require_int(values, "padding"));
            layer->set_weights(load_weights_from_file((base_dir / require_string(values, "weights")).string(),
                                                      static_cast<std::size_t>(layer->expected_weight_count())));
            layer->set_bias(load_weights_from_file((base_dir / require_string(values, "bias")).string(),
                                                   static_cast<std::size_t>(layer->expected_bias_count())));
            model.add(std::move(layer));
        } else if (layer_type == "ReLU") {
            model.add(std::make_unique<ReLU>());
        } else if (layer_type == "MaxPool2D") {
            model.add(std::make_unique<MaxPool2D>(require_int(values, "kernel_size"),
                                                  require_int(values, "stride")));
        } else if (layer_type == "Flatten") {
            model.add(std::make_unique<Flatten>());
        } else if (layer_type == "Linear") {
            auto layer = std::make_unique<Linear>(require_int(values, "in_features"),
                                                  require_int(values, "out_features"));
            layer->set_weights(load_weights_from_file((base_dir / require_string(values, "weights")).string(),
                                                      static_cast<std::size_t>(layer->expected_weight_count())));
            layer->set_bias(load_weights_from_file((base_dir / require_string(values, "bias")).string(),
                                                   static_cast<std::size_t>(layer->expected_bias_count())));
            model.add(std::move(layer));
        } else if (layer_type == "Softmax") {
            model.add(std::make_unique<Softmax>());
        } else {
            throw std::runtime_error("Unsupported layer type in manifest: " + layer_type);
        }
    }

    return model;
}
