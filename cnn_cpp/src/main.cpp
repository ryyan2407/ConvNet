#include <filesystem>
#include <iostream>
#include <memory>

#include "conv2d.hpp"
#include "flatten.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "softmax.hpp"
#include "weights_loader.hpp"
#include "project_config.hpp"

int main() {
    try {
        const std::filesystem::path project_root = CNN_CPP_PROJECT_ROOT;
        const std::filesystem::path input_path = project_root / "data" / "sample_input.txt";
        const std::filesystem::path weights_root = project_root / "weights";

        Tensor input = load_image_as_tensor(input_path.string(), 1, 8, 8, true);

        auto conv1 = std::make_unique<Conv2D>(1, 2, 3, 1, 1);
        conv1->set_weights(load_weights_from_file((weights_root / "conv1_weights.txt").string(),
                                                  static_cast<std::size_t>(conv1->expected_weight_count())));
        conv1->set_bias(load_weights_from_file((weights_root / "conv1_bias.txt").string(),
                                               static_cast<std::size_t>(conv1->expected_bias_count())));

        auto conv2 = std::make_unique<Conv2D>(2, 3, 3, 1, 1);
        conv2->set_weights(load_weights_from_file((weights_root / "conv2_weights.txt").string(),
                                                  static_cast<std::size_t>(conv2->expected_weight_count())));
        conv2->set_bias(load_weights_from_file((weights_root / "conv2_bias.txt").string(),
                                               static_cast<std::size_t>(conv2->expected_bias_count())));

        auto linear = std::make_unique<Linear>(12, 3);
        linear->set_weights(load_weights_from_file((weights_root / "fc_weights.txt").string(),
                                                   static_cast<std::size_t>(linear->expected_weight_count())));
        linear->set_bias(load_weights_from_file((weights_root / "fc_bias.txt").string(),
                                                static_cast<std::size_t>(linear->expected_bias_count())));

        Sequential model;
        model.add(std::move(conv1));
        model.add(std::make_unique<ReLU>());
        model.add(std::make_unique<MaxPool2D>(2, 2));
        model.add(std::move(conv2));
        model.add(std::make_unique<ReLU>());
        model.add(std::make_unique<MaxPool2D>(2, 2));
        model.add(std::make_unique<Flatten>());
        model.add(std::move(linear));
        model.add(std::make_unique<Softmax>());

        Tensor output = model.forward(input);

        std::cout << "Input shape: ";
        input.print_shape();
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
