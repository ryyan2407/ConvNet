#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "image_loader.hpp"
#include "model_config.hpp"
#include "model_io.hpp"
#include "project_config.hpp"

int main(int argc, char** argv) {
    try {
        const std::filesystem::path project_root = CNN_CPP_PROJECT_ROOT;
        const std::filesystem::path input_path = project_root / "data" / "sample_input.txt";
        const std::filesystem::path model_path =
            argc >= 2 ? std::filesystem::path(argv[1]) : project_root / "weights" / "sample_infer_model.txt";

        Tensor input = load_image_as_tensor(input_path.string(), 1, 8, 8, true);
        std::ifstream manifest(model_path);
        if (!manifest) {
            throw std::runtime_error("Failed to open model path: " + model_path.string());
        }
        std::string header;
        std::getline(manifest, header);
        Sequential model;
        if (header == "cnn_cpp_model_v1") {
            model = load_model_artifact(model_path.string());
        } else if (header == "cnn_cpp_config_v1") {
            model = build_model_from_config(model_path.string());
        } else {
            throw std::runtime_error("Unknown model file header: " + header);
        }

        Tensor output = model.forward(input);

        std::cout << "Input shape: ";
        input.print_shape();
        std::cout << "Model file: " << model_path << "\n";
        std::cout << "Output shape: ";
        output.print_shape();

        int predicted_class = 0;
        float best_probability = output(0, 0, 0, 0);
        std::cout << "Probabilities:\n";
        for (int i = 0; i < output.W(); ++i) {
            const float probability = output(0, 0, 0, i);
            std::cout << "class " << i << ": " << probability << "\n";
            if (probability > best_probability) {
                best_probability = probability;
                predicted_class = i;
            }
        }

        std::cout << "Predicted class: " << predicted_class << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << "\n";
        return 1;
    }
}
