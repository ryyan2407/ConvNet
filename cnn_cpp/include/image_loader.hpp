#pragma once

#include <string>
#include <vector>

#include "tensor.hpp"

struct LabeledDataset {
    Tensor data;
    std::vector<int> labels;
};

Tensor load_image_as_tensor(const std::string& path, int channels, int height, int width, bool normalize = true);
LabeledDataset load_labeled_dataset(const std::string& images_path,
                                    const std::string& labels_path,
                                    int channels,
                                    int height,
                                    int width,
                                    bool normalize = true);
LabeledDataset load_idx_dataset(const std::string& images_path,
                                const std::string& labels_path,
                                bool normalize = true,
                                int max_samples = -1);
