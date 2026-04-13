#include "checkpoint_io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::unordered_map<std::string, std::string> parse_key_values(std::istream& stream) {
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::size_t pos = line.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid training state line: " + line);
        }
        values[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return values;
}

}  // namespace

void save_training_state(const std::string& path, const TrainingState& state) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open training state for writing: " + path);
    }

    file << "cnn_cpp_training_state_v1\n";
    file << "epoch_completed=" << state.epoch_completed << "\n";
    file << "best_metric=" << state.best_metric << "\n";
    file << "model_config_path=" << state.model_config_path << "\n";
}

TrainingState load_training_state(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open training state for reading: " + path);
    }

    std::string header;
    std::getline(file, header);
    if (header != "cnn_cpp_training_state_v1") {
        throw std::runtime_error("Unsupported training state header: " + header);
    }

    const auto values = parse_key_values(file);
    TrainingState state;
    if (const auto it = values.find("epoch_completed"); it != values.end()) {
        state.epoch_completed = std::stoi(it->second);
    }
    if (const auto it = values.find("best_metric"); it != values.end()) {
        state.best_metric = std::stof(it->second);
    }
    if (const auto it = values.find("model_config_path"); it != values.end()) {
        state.model_config_path = it->second;
    }
    return state;
}
