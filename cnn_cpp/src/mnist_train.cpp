#include <algorithm>
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

#include "conv2d.hpp"
#include "cross_entropy.hpp"
#include "flatten.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "sgd.hpp"
#include "softmax.hpp"
#include "weights_loader.hpp"

namespace {

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

int count_correct(const Tensor& logits, LabelView labels) {
    const float* logits_data = logits.raw_data();
    int correct = 0;
    for (int n = 0; n < logits.N(); ++n) {
        const int base = logits.offset_unchecked(n, 0, 0, 0);
        int best_index = 0;
        float best_value = logits_data[base];
        for (int i = 1; i < logits.W(); ++i) {
            if (logits_data[base + i] > best_value) {
                best_value = logits_data[base + i];
                best_index = i;
            }
        }
        if (best_index == labels[n]) {
            ++correct;
        }
    }
    return correct;
}

void save_model_checkpoint(const std::filesystem::path& weights_dir,
                           const Conv2D& conv1,
                           const Conv2D& conv2,
                           const Linear& linear) {
    std::filesystem::create_directories(weights_dir);
    save_weights_to_file((weights_dir / "conv1_weights.txt").string(), conv1.weights());
    save_weights_to_file((weights_dir / "conv1_bias.txt").string(), conv1.bias());
    save_weights_to_file((weights_dir / "conv2_weights.txt").string(), conv2.weights());
    save_weights_to_file((weights_dir / "conv2_bias.txt").string(), conv2.bias());
    save_weights_to_file((weights_dir / "fc_weights.txt").string(), linear.weights());
    save_weights_to_file((weights_dir / "fc_bias.txt").string(), linear.bias());
}

std::string format_metric(float value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4) << value;
    return stream.str();
}

LabelView make_range_labels(const std::vector<int>& labels, std::size_t start, std::size_t end) {
    return LabelView{labels.data() + static_cast<std::ptrdiff_t>(start), static_cast<int>(end - start)};
}

std::pair<float, float> evaluate_dataset(Sequential& model,
                                         const LabeledDataset& dataset,
                                         CrossEntropyLoss& loss,
                                         int batch_size) {
    float total_loss = 0.0f;
    int correct = 0;
    std::size_t total_seen = 0;
#if defined(CNN_CPP_USE_OPENMP)
#pragma omp parallel
#endif
    {
        CrossEntropyLoss local_loss = loss;
        float thread_loss = 0.0f;
        int thread_correct = 0;
        std::size_t thread_seen = 0;

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp for schedule(static)
#endif
        for (std::int64_t start = 0; start < static_cast<std::int64_t>(dataset.data.N());
             start += static_cast<std::int64_t>(batch_size)) {
            const std::size_t batch_start = static_cast<std::size_t>(start);
            const std::size_t end = std::min(batch_start + static_cast<std::size_t>(batch_size),
                                             static_cast<std::size_t>(dataset.data.N()));
            Tensor batch = dataset.data.slice_n(static_cast<int>(batch_start), static_cast<int>(end - batch_start));
            LabelView labels = make_range_labels(dataset.labels, batch_start, end);
            Tensor logits = model.predict(batch);
            thread_loss += local_loss.forward(logits, labels) * static_cast<float>(labels.size);
            thread_correct += count_correct(logits, labels);
            thread_seen += static_cast<std::size_t>(labels.size);
        }

#if defined(CNN_CPP_USE_OPENMP)
#pragma omp critical
#endif
        {
            total_loss += thread_loss;
            correct += thread_correct;
            total_seen += thread_seen;
        }
    }

    return {
        total_loss / static_cast<float>(total_seen),
        static_cast<float>(correct) / static_cast<float>(total_seen)
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: ./cnn_mnist_train <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [max_samples] [epochs] [batch_size] [test-images.idx3-ubyte] [test-labels.idx1-ubyte] [checkpoint_dir]\n";
            return 1;
        }

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

        LabeledDataset test_dataset;
        const bool has_test_set = !test_images_path.empty() && !test_labels_path.empty();
        if (has_test_set) {
            test_dataset = load_idx_dataset(test_images_path.string(), test_labels_path.string(), true, max_samples);
        }

        if (dataset.data.H() != 28 || dataset.data.W() != 28) {
            throw std::runtime_error("cnn_mnist_train expects 28x28 IDX images");
        }

        auto conv1 = std::make_unique<Conv2D>(1, 8, 3, 1, 1);
        auto* conv1_ptr = conv1.get();
        auto conv2 = std::make_unique<Conv2D>(8, 16, 3, 1, 1);
        auto* conv2_ptr = conv2.get();
        auto linear = std::make_unique<Linear>(16 * 7 * 7, 10);
        auto* linear_ptr = linear.get();

        Sequential model;
        model.add(std::move(conv1));
        model.add(std::make_unique<ReLU>());
        model.add(std::make_unique<MaxPool2D>(2, 2));
        model.add(std::move(conv2));
        model.add(std::make_unique<ReLU>());
        model.add(std::make_unique<MaxPool2D>(2, 2));
        model.add(std::make_unique<Flatten>());
        model.add(std::move(linear));

        CrossEntropyLoss loss;
        SGD optimizer(0.01f);
        std::mt19937 generator(42);
        const std::size_t num_batches =
            (dataset.data.N() + batch_size - 1) / static_cast<std::size_t>(batch_size);
        std::vector<std::size_t> batch_order(num_batches);
        for (std::size_t i = 0; i < batch_order.size(); ++i) {
            batch_order.at(i) = i;
        }

        float best_metric = -1.0f;

        for (int epoch = 0; epoch < epochs; ++epoch) {
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
                save_model_checkpoint(best_dir, *conv1_ptr, *conv2_ptr, *linear_ptr);

                const std::filesystem::path epoch_dir =
                    checkpoint_root / ("epoch_" + std::to_string(epoch + 1) + "_metric_" + format_metric(checkpoint_metric));
                save_model_checkpoint(epoch_dir, *conv1_ptr, *conv2_ptr, *linear_ptr);
            }
        }

        const std::filesystem::path weights_dir = checkpoint_root / "final";
        save_model_checkpoint(weights_dir, *conv1_ptr, *conv2_ptr, *linear_ptr);

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
