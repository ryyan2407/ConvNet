#pragma once

#include <vector>

#include "loss.hpp"

class CrossEntropyLoss : public Loss {
public:
    float forward(const Tensor& logits, LabelView targets) override;
    float forward(const Tensor& logits, const std::vector<int>& targets) override;
    Tensor backward() override;

private:
    Tensor cached_probabilities_;
    std::vector<int> cached_targets_;
};
