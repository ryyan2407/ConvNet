#pragma once

#include <string>

struct TrainingState {
    int epoch_completed = 0;
    float best_metric = -1.0f;
    std::string model_config_path;
};

void save_training_state(const std::string& path, const TrainingState& state);
TrainingState load_training_state(const std::string& path);
