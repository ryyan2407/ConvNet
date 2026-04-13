#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "conv2d.hpp"
#include "cross_entropy.hpp"
#include "flatten.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "model_io.hpp"
#include "project_config.hpp"
#include "relu.hpp"
#include "sgd.hpp"
#include "softmax.hpp"
#include "weights_loader.hpp"

int argmax(const Tensor& output) {
    int best_index = 0;
    float best_value = output(0, 0, 0, 0);
    for (int i = 1; i < output.W(); ++i) {
        if (output(0, 0, 0, i) > best_value) {
            best_value = output(0, 0, 0, i);
            best_index = i;
        }
    }
    return best_index;
}

int main() {
    try {
        const std::filesystem::path project_root = CNN_CPP_PROJECT_ROOT;
        const std::filesystem::path trained_root = project_root / "weights" / "trained_demo";
        std::filesystem::create_directories(trained_root);
        const std::filesystem::path images_path = project_root / "data" / "demo_train_images.txt";
        const std::filesystem::path labels_path = project_root / "data" / "demo_train_labels.txt";
        const LabeledDataset dataset = load_labeled_dataset(images_path.string(), labels_path.string(), 1, 4, 4, true);

        auto conv = std::make_unique<Conv2D>(1, 2, 3, 1, 1);
        auto linear = std::make_unique<Linear>(8, 2);

        Sequential model;
        model.add(std::move(conv));
        model.add(std::make_unique<ReLU>());
        model.add(std::make_unique<MaxPool2D>(2, 2));
        model.add(std::make_unique<Flatten>());
        model.add(std::move(linear));

        CrossEntropyLoss loss;
        SGD optimizer(0.1f);

        const int epochs = 120;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int correct = 0;
            for (int i = 0; i < dataset.data.N(); ++i) {
                optimizer.zero_grad(model);
                Tensor sample = dataset.data.slice_n(i, 1);
                Tensor logits = model.forward(sample);
                const int target = dataset.labels.at(static_cast<std::size_t>(i));
                epoch_loss += loss.forward(logits, LabelView{&target, 1});
                Tensor grad = loss.backward();
                model.backward(grad);
                optimizer.step(model);
                if (argmax(logits) == dataset.labels.at(static_cast<std::size_t>(i))) {
                    ++correct;
                }
            }

            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                std::cout << "epoch " << (epoch + 1)
                          << " loss=" << (epoch_loss / static_cast<float>(dataset.data.N()))
                          << " accuracy=" << (static_cast<float>(correct) / static_cast<float>(dataset.data.N()))
                          << "\n";
            }
        }

        save_model_artifact(model, trained_root.string());
        Sequential reloaded_model = load_model_artifact((trained_root / "model.txt").string());

        Softmax softmax;
        std::cout << "Saved trained weights to: " << trained_root << "\n";
        std::cout << "Loaded dataset: " << dataset.data.N() << " samples from " << images_path << "\n";
        std::cout << "Final predictions:\n";
        for (int i = 0; i < dataset.data.N(); ++i) {
            Tensor sample = dataset.data.slice_n(i, 1);
            Tensor probabilities = softmax.forward(model.forward(sample));
            Tensor reloaded_probabilities = softmax.forward(reloaded_model.forward(sample));
            std::cout << "sample " << i << " label=" << dataset.labels.at(static_cast<std::size_t>(i))
                      << " pred=" << argmax(probabilities)
                      << " p0=" << probabilities(0, 0, 0, 0)
                      << " p1=" << probabilities(0, 0, 0, 1)
                      << " reload_pred=" << argmax(reloaded_probabilities) << "\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Training demo failed: " << error.what() << "\n";
        return 1;
    }
}
