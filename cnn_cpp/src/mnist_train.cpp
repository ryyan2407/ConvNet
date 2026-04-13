#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "checkpoint_io.hpp"
#include "conv2d.hpp"
#include "cross_entropy.hpp"
#include "eval_utils.hpp"
#include "flatten.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "model_config.hpp"
#include "model_io.hpp"
#include "project_config.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "sgd.hpp"
#include "softmax.hpp"

namespace {

std::string format_metric(float value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4) << value;
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: ./cnn_mnist_train <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [max_samples] [epochs] [batch_size] [test-images.idx3-ubyte] [test-labels.idx1-ubyte] [checkpoint_dir] [model_config] [resume_artifact_dir]\n";
            return 1;
        }

        const std::filesystem::path project_root = CNN_CPP_PROJECT_ROOT;
        const std::string images_path = argv[1];
        const std::string labels_path = argv[2];
        const int max_samples = argc >= 4 ? std::stoi(argv[3]) : 1000;
        const int epochs = argc >= 5 ? std::stoi(argv[4]) : 3;
        const int batch_size = argc >= 6 ? std::stoi(argv[5]) : 32;
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }

        const LabeledDataset dataset = load_idx_dataset(images_path, labels_path, true, max_samples);
        std::filesystem::path test_images_path;
        std::filesystem::path test_labels_path;
        std::filesystem::path checkpoint_root = "mnist_trained_weights";
        if (argc >= 8) {
            test_images_path = argv[6];
            test_labels_path = argv[7];
        } else {
            const std::filesystem::path train_images_fs = images_path;
            const std::filesystem::path train_labels_fs = labels_path;
            const std::filesystem::path base_dir = train_images_fs.parent_path();
            const std::filesystem::path auto_test_images = base_dir / "t10k-images.idx3-ubyte";
            const std::filesystem::path auto_test_labels = base_dir / "t10k-labels.idx1-ubyte";
            if (std::filesystem::exists(auto_test_images) && std::filesystem::exists(auto_test_labels)) {
                test_images_path = auto_test_images;
                test_labels_path = auto_test_labels;
            } else if (train_images_fs.filename() == "train-images.idx3-ubyte" &&
                       train_labels_fs.filename() == "train-labels.idx1-ubyte") {
                const std::filesystem::path cwd_auto_test_images = "t10k-images.idx3-ubyte";
                const std::filesystem::path cwd_auto_test_labels = "t10k-labels.idx1-ubyte";
                if (std::filesystem::exists(cwd_auto_test_images) && std::filesystem::exists(cwd_auto_test_labels)) {
                    test_images_path = cwd_auto_test_images;
                    test_labels_path = cwd_auto_test_labels;
                }
            }
        }
        if (argc >= 9) {
            checkpoint_root = argv[8];
        }
        std::filesystem::path model_config_path = project_root / "configs" / "mnist_cnn.txt";
        if (argc >= 10) {
            model_config_path = argv[9];
        }
        std::filesystem::path resume_artifact_dir;
        if (argc >= 11) {
            resume_artifact_dir = argv[10];
        }

        LabeledDataset test_dataset;
        const bool has_test_set = !test_images_path.empty() && !test_labels_path.empty();
        if (has_test_set) {
            test_dataset = load_idx_dataset(test_images_path.string(), test_labels_path.string(), true, max_samples);
        }

        if (dataset.data.H() != 28 || dataset.data.W() != 28) {
            throw std::runtime_error("cnn_mnist_train expects 28x28 IDX images");
        }

        Sequential model;
        TrainingState state;
        if (!resume_artifact_dir.empty()) {
            model = load_model_artifact((resume_artifact_dir / "model.txt").string());
            const std::filesystem::path state_path = resume_artifact_dir / "training_state.txt";
            if (std::filesystem::exists(state_path)) {
                state = load_training_state(state_path.string());
                if (!state.model_config_path.empty()) {
                    model_config_path = state.model_config_path;
                }
            }
            std::cout << "Resuming from: " << resume_artifact_dir << "\n";
            std::cout << "Starting after epoch " << state.epoch_completed << "\n";
        } else {
            model = build_model_from_config(model_config_path.string());
        }

        CrossEntropyLoss loss;
        SGD optimizer(0.01f);
        std::mt19937 generator(42);
        const std::size_t num_batches =
            (dataset.data.N() + batch_size - 1) / static_cast<std::size_t>(batch_size);
        std::vector<std::size_t> batch_order(num_batches);
        for (std::size_t i = 0; i < batch_order.size(); ++i) {
            batch_order.at(i) = i;
        }

        float best_metric = state.best_metric;
        const int start_epoch = state.epoch_completed;

        for (int epoch_offset = 0; epoch_offset < epochs; ++epoch_offset) {
            const int epoch = start_epoch + epoch_offset;
            std::shuffle(batch_order.begin(), batch_order.end(), generator);
            float epoch_loss = 0.0f;
            int correct = 0;
            std::size_t total_seen = 0;
            for (std::size_t batch_id : batch_order) {
                const std::size_t start = batch_id * static_cast<std::size_t>(batch_size);
                const std::size_t end = std::min(start + static_cast<std::size_t>(batch_size),
                                                 static_cast<std::size_t>(dataset.data.N()));
                optimizer.zero_grad(model);
                Tensor batch = dataset.data.slice_n(static_cast<int>(start), static_cast<int>(end - start));
                LabelView labels = make_range_labels(dataset.labels, start, end);
                Tensor logits = model.forward(batch);
                epoch_loss += loss.forward(logits, labels) * static_cast<float>(labels.size);
                model.backward(loss.backward());
                optimizer.step(model);
                correct += count_correct(logits, labels);
                total_seen += static_cast<std::size_t>(labels.size);
            }

            const float train_loss = epoch_loss / static_cast<float>(total_seen);
            const float train_accuracy = static_cast<float>(correct) / static_cast<float>(total_seen);
            std::cout << "epoch " << (epoch + 1)
                      << " loss=" << train_loss
                      << " accuracy=" << train_accuracy;
            float checkpoint_metric = train_accuracy;
            if (has_test_set) {
                const auto [test_loss, test_accuracy] = evaluate_dataset(model, test_dataset, loss, batch_size);
                std::cout << " test_loss=" << test_loss
                          << " test_accuracy=" << test_accuracy;
                checkpoint_metric = test_accuracy;
            }
            std::cout << "\n";

            if (checkpoint_metric > best_metric) {
                best_metric = checkpoint_metric;
                const std::filesystem::path best_dir = checkpoint_root / "best";
                save_model_artifact(model, best_dir.string());
                save_training_state((best_dir / "training_state.txt").string(),
                                    TrainingState{epoch + 1, best_metric, model_config_path.string()});

                const std::filesystem::path epoch_dir =
                    checkpoint_root / ("epoch_" + std::to_string(epoch + 1) + "_metric_" + format_metric(checkpoint_metric));
                save_model_artifact(model, epoch_dir.string());
                save_training_state((epoch_dir / "training_state.txt").string(),
                                    TrainingState{epoch + 1, best_metric, model_config_path.string()});
            }
        }

        const std::filesystem::path weights_dir = checkpoint_root / "final";
        save_model_artifact(model, weights_dir.string());
        save_training_state((weights_dir / "training_state.txt").string(),
                            TrainingState{start_epoch + epochs, best_metric, model_config_path.string()});

        Softmax softmax;
        Tensor first_train_sample = dataset.data.slice_n(0, 1);
        Tensor probabilities = softmax.infer(model.predict(first_train_sample));
        std::cout << "sample0 label=" << dataset.labels.at(0)
                  << " pred=" << argmax(probabilities)
                  << " p=" << probabilities(0, 0, 0, argmax(probabilities)) << "\n";
        if (has_test_set) {
            Tensor first_test_sample = test_dataset.data.slice_n(0, 1);
            Tensor test_probabilities = softmax.infer(model.predict(first_test_sample));
            std::cout << "test_sample0 label=" << test_dataset.labels.at(0)
                      << " pred=" << argmax(test_probabilities)
                      << " p=" << test_probabilities(0, 0, 0, argmax(test_probabilities)) << "\n";
        }
        std::cout << "Saved final weights to: " << weights_dir << "\n";
        std::cout << "Saved best checkpoint to: " << (checkpoint_root / "best") << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MNIST training failed: " << error.what() << "\n";
        return 1;
    }
}
