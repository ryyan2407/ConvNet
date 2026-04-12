#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "image_loader.hpp"
#include "project_config.hpp"
#include "weights_loader.hpp"

namespace {

void write_be_u32(std::ofstream& stream, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xFF),
        static_cast<unsigned char>((value >> 16) & 0xFF),
        static_cast<unsigned char>((value >> 8) & 0xFF),
        static_cast<unsigned char>(value & 0xFF)
    };
    stream.write(reinterpret_cast<const char*>(bytes), 4);
}

}  // namespace

int main() {
    const std::filesystem::path project_root = CNN_CPP_PROJECT_ROOT;

    Tensor input = load_image_as_tensor((project_root / "data" / "sample_input.txt").string(), 1, 8, 8, true);
    assert(input.N() == 1);
    assert(input.C() == 1);
    assert(input.H() == 8);
    assert(input.W() == 8);
    assert(input(0, 0, 0, 0) == 0.0f);
    assert(input(0, 0, 7, 7) == 0.0f);

    auto conv1_weights = load_weights_from_file((project_root / "weights" / "conv1_weights.txt").string(), 18);
    assert(conv1_weights.size() == 18);

    const LabeledDataset dataset = load_labeled_dataset(
        (project_root / "data" / "demo_train_images.txt").string(),
        (project_root / "data" / "demo_train_labels.txt").string(),
        1, 4, 4, true);
    assert(dataset.labels.size() == 8);
    assert(dataset.data.N() == 8);
    assert(dataset.data.C() == 1);
    assert(dataset.data.H() == 4);
    assert(dataset.data.W() == 4);
    assert(dataset.labels.at(0) == 0);
    assert(dataset.labels.at(7) == 1);
    assert(dataset.data(0, 0, 0, 0) == 1.0f);
    assert(dataset.data(0, 0, 0, 1) == 0.0f);
    assert(dataset.data(7, 0, 0, 0) == 0.0f);

    const std::filesystem::path idx_images = project_root / "data" / "test_images.idx3-ubyte";
    const std::filesystem::path idx_labels = project_root / "data" / "test_labels.idx1-ubyte";
    {
        std::ofstream image_file(idx_images, std::ios::binary);
        std::ofstream label_file(idx_labels, std::ios::binary);
        write_be_u32(image_file, 2051);
        write_be_u32(image_file, 2);
        write_be_u32(image_file, 2);
        write_be_u32(image_file, 2);
        const unsigned char image_bytes[8] = {0, 255, 128, 64, 255, 0, 64, 128};
        image_file.write(reinterpret_cast<const char*>(image_bytes), 8);

        write_be_u32(label_file, 2049);
        write_be_u32(label_file, 2);
        const unsigned char label_bytes[2] = {3, 7};
        label_file.write(reinterpret_cast<const char*>(label_bytes), 2);
    }

    const LabeledDataset idx_dataset = load_idx_dataset(idx_images.string(), idx_labels.string(), true, -1);
    assert(idx_dataset.data.N() == 2);
    assert(idx_dataset.labels.at(0) == 3);
    assert(idx_dataset.labels.at(1) == 7);
    assert(idx_dataset.data.H() == 2);
    assert(idx_dataset.data.W() == 2);
    assert(idx_dataset.data(0, 0, 0, 0) == 0.0f);
    assert(idx_dataset.data(0, 0, 0, 1) == 1.0f);
    assert(idx_dataset.data(1, 0, 0, 0) == 1.0f);

    const std::filesystem::path temp_path = project_root / "weights" / "test_roundtrip.txt";
    const std::vector<float> roundtrip_values = {1.25f, -2.5f, 3.75f};
    save_weights_to_file(temp_path.string(), roundtrip_values);
    const auto loaded_roundtrip = load_weights_from_file(temp_path.string(), roundtrip_values.size());
    assert(loaded_roundtrip == roundtrip_values);

    bool threw = false;
    try {
        (void)load_weights_from_file((project_root / "weights" / "conv1_bias.txt").string(), 3);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    return 0;
}
