#include <filesystem>
#include <iostream>
#include <string>

#include "cross_entropy.hpp"
#include "eval_utils.hpp"
#include "image_loader.hpp"
#include "model_io.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: ./cnn_eval <model_artifact_manifest> <images.idx3-ubyte> <labels.idx1-ubyte> [max_samples] [batch_size]\n";
            return 1;
        }

        const std::string model_manifest = argv[1];
        const std::string images_path = argv[2];
        const std::string labels_path = argv[3];
        const int max_samples = argc >= 5 ? std::stoi(argv[4]) : -1;
        const int batch_size = argc >= 6 ? std::stoi(argv[5]) : 32;
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }

        Sequential model = load_model_artifact(model_manifest);
        const LabeledDataset dataset = load_idx_dataset(images_path, labels_path, true, max_samples);
        CrossEntropyLoss loss;
        const auto [dataset_loss, dataset_accuracy] = evaluate_dataset(model, dataset, loss, batch_size);

        std::cout << "Evaluated model: " << model_manifest << "\n";
        std::cout << "Samples: " << dataset.data.N() << "\n";
        std::cout << "Loss: " << dataset_loss << "\n";
        std::cout << "Accuracy: " << dataset_accuracy << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Evaluation failed: " << error.what() << "\n";
        return 1;
    }
}
