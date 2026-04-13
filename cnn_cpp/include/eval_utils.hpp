#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "cross_entropy.hpp"
#include "image_loader.hpp"
#include "sequential.hpp"

int argmax(const Tensor& output);
int count_correct(const Tensor& logits, LabelView labels);
LabelView make_range_labels(const std::vector<int>& labels, std::size_t start, std::size_t end);
std::pair<float, float> evaluate_dataset(Sequential& model,
                                         const LabeledDataset& dataset,
                                         CrossEntropyLoss& loss,
                                         int batch_size);
