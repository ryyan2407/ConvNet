#include "image_loader.hpp"

#include <fstream>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

namespace {

std::uint32_t read_be_u32(std::ifstream& stream, const std::string& path) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (!stream) {
        throw std::runtime_error("Failed to read IDX header from: " + path);
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

}  // namespace

Tensor load_image_as_tensor(const std::string& path, int channels, int height, int width, bool normalize) {
    const auto values = read_floats_from_file(path);
    const std::size_t expected_size = static_cast<std::size_t>(channels * height * width);
    assert_expected_size(values.size(), expected_size, path);

    Tensor tensor(1, channels, height, width);
    std::size_t index = 0;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float value = values.at(index++);
                if (normalize) {
                    value /= 255.0f;
                }
                tensor(0, c, h, w) = value;
            }
        }
    }
    return tensor;
}

LabeledDataset load_labeled_dataset(const std::string& images_path,
                                    const std::string& labels_path,
                                    int channels,
                                    int height,
                                    int width,
                                    bool normalize) {
    const int pixels_per_sample = channels * height * width;
    if (pixels_per_sample <= 0) {
        throw std::runtime_error("Dataset dimensions must be positive");
    }

    std::ifstream images_file(images_path);
    if (!images_file) {
        throw std::runtime_error("Failed to open dataset images file: " + images_path);
    }

    std::ifstream labels_file(labels_path);
    if (!labels_file) {
        throw std::runtime_error("Failed to open dataset labels file: " + labels_path);
    }

    LabeledDataset dataset;
    std::vector<std::vector<float>> image_rows;
    std::string image_line;
    std::string label_line;
    int line_number = 0;
    while (std::getline(images_file, image_line)) {
        ++line_number;
        if (image_line.empty()) {
            continue;
        }

        if (!std::getline(labels_file, label_line)) {
            throw std::runtime_error("Dataset labels ended early at sample " + std::to_string(line_number));
        }

        std::stringstream image_stream(image_line);
        std::vector<float> pixels;
        pixels.reserve(static_cast<std::size_t>(pixels_per_sample));
        float pixel = 0.0f;
        while (image_stream >> pixel) {
            pixels.push_back(pixel);
        }
        assert_expected_size(pixels.size(), static_cast<std::size_t>(pixels_per_sample),
                             "Dataset sample " + std::to_string(line_number));

        std::stringstream label_stream(label_line);
        int label = 0;
        if (!(label_stream >> label)) {
            throw std::runtime_error("Invalid label at sample " + std::to_string(line_number));
        }

        image_rows.push_back(std::move(pixels));
        dataset.labels.push_back(label);
    }

    if (dataset.labels.empty()) {
        throw std::runtime_error("Dataset is empty: " + images_path);
    }

    if (std::getline(labels_file, label_line)) {
        throw std::runtime_error("Dataset labels contain more entries than images");
    }

    dataset.data = Tensor(static_cast<int>(dataset.labels.size()), channels, height, width);
    float* dataset_data = dataset.data.raw_data();
    const std::size_t sample_size = static_cast<std::size_t>(channels * height * width);
    for (std::size_t sample_index = 0; sample_index < image_rows.size(); ++sample_index) {
        const std::vector<float>& pixels = image_rows.at(sample_index);
        const std::size_t base = sample_index * sample_size;
        for (std::size_t i = 0; i < sample_size; ++i) {
            float value = pixels.at(i);
            if (normalize) {
                value /= 255.0f;
            }
            dataset_data[base + i] = value;
        }
    }
    return dataset;
}

LabeledDataset load_idx_dataset(const std::string& images_path,
                                const std::string& labels_path,
                                bool normalize,
                                int max_samples) {
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file) {
        throw std::runtime_error("Failed to open IDX images file: " + images_path);
    }

    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file) {
        throw std::runtime_error("Failed to open IDX labels file: " + labels_path);
    }

    const std::uint32_t image_magic = read_be_u32(images_file, images_path);
    const std::uint32_t image_count = read_be_u32(images_file, images_path);
    const std::uint32_t rows = read_be_u32(images_file, images_path);
    const std::uint32_t cols = read_be_u32(images_file, images_path);
    if (image_magic != 2051) {
        throw std::runtime_error("Invalid IDX image magic number in: " + images_path);
    }

    const std::uint32_t label_magic = read_be_u32(labels_file, labels_path);
    const std::uint32_t label_count = read_be_u32(labels_file, labels_path);
    if (label_magic != 2049) {
        throw std::runtime_error("Invalid IDX label magic number in: " + labels_path);
    }

    if (image_count != label_count) {
        throw std::runtime_error("IDX image/label count mismatch");
    }

    const std::uint32_t sample_count =
        max_samples > 0 ? std::min(image_count, static_cast<std::uint32_t>(max_samples)) : image_count;
    const std::uint32_t pixel_count = rows * cols;
    if (pixel_count == 0) {
        throw std::runtime_error("IDX dataset has invalid image dimensions");
    }

    LabeledDataset dataset;
    dataset.labels.reserve(sample_count);
    dataset.data = Tensor(static_cast<int>(sample_count), 1, static_cast<int>(rows), static_cast<int>(cols));
    float* dataset_data = dataset.data.raw_data();

    std::vector<unsigned char> image_bytes(pixel_count);
    for (std::uint32_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        images_file.read(reinterpret_cast<char*>(image_bytes.data()), static_cast<std::streamsize>(image_bytes.size()));
        if (!images_file) {
            throw std::runtime_error("Unexpected end of IDX images file: " + images_path);
        }

        unsigned char label_byte = 0;
        labels_file.read(reinterpret_cast<char*>(&label_byte), 1);
        if (!labels_file) {
            throw std::runtime_error("Unexpected end of IDX labels file: " + labels_path);
        }

        const std::size_t base = static_cast<std::size_t>(sample_index) * static_cast<std::size_t>(pixel_count);
        for (std::size_t i = 0; i < pixel_count; ++i) {
            float value = static_cast<float>(image_bytes.at(i));
            if (normalize) {
                value /= 255.0f;
            }
            dataset_data[base + i] = value;
        }
        dataset.labels.push_back(static_cast<int>(label_byte));
    }

    if (dataset.labels.empty()) {
        throw std::runtime_error("IDX dataset is empty");
    }

    return dataset;
}
