#include <cassert>
#include <filesystem>

#include "checkpoint_io.hpp"

int main() {
    const std::filesystem::path path = "checkpoint_state_roundtrip.txt";
    const TrainingState original{7, 0.8125f, "configs/mnist_cnn.txt"};
    save_training_state(path.string(), original);
    const TrainingState loaded = load_training_state(path.string());

    assert(loaded.epoch_completed == original.epoch_completed);
    assert(loaded.best_metric == original.best_metric);
    assert(loaded.model_config_path == original.model_config_path);
    return 0;
}
