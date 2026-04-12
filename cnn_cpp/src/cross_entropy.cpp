#include "cross_entropy.hpp"

#include <cmath>
#include <stdexcept>

float CrossEntropyLoss::forward(const Tensor& logits, LabelView targets) {
    if (logits.C() != 1 || logits.H() != 1) {
        throw std::runtime_error("CrossEntropyLoss expects logits shape [N, 1, 1, classes]");
    }
    if (targets.size != logits.N()) {
        throw std::runtime_error("CrossEntropyLoss target size mismatch");
    }

    cached_targets_.assign(targets.data, targets.data + targets.size);
    cached_probabilities_ = Tensor(logits.N(), 1, 1, logits.W());

    float loss = 0.0f;
    const float* logits_data = logits.raw_data();
    float* probability_data = cached_probabilities_.raw_data();
    for (int n = 0; n < logits.N(); ++n) {
        const int target = targets[n];
        if (target < 0 || target >= logits.W()) {
            throw std::runtime_error("CrossEntropyLoss target out of range");
        }

        const int base = logits.offset_unchecked(n, 0, 0, 0);
        float max_logit = logits_data[base];
        for (int i = 1; i < logits.W(); ++i) {
            if (logits_data[base + i] > max_logit) {
                max_logit = logits_data[base + i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < logits.W(); ++i) {
            const float probability = std::exp(logits_data[base + i] - max_logit);
            probability_data[base + i] = probability;
            sum_exp += probability;
        }

        for (int i = 0; i < logits.W(); ++i) {
            probability_data[base + i] /= sum_exp;
        }

        const float target_probability = probability_data[base + target];
        loss -= std::log(target_probability + 1e-12f);
    }

    return loss / static_cast<float>(logits.N());
}

float CrossEntropyLoss::forward(const Tensor& logits, const std::vector<int>& targets) {
    return forward(logits, LabelView{targets.data(), static_cast<int>(targets.size())});
}

Tensor CrossEntropyLoss::backward() {
    Tensor gradient = cached_probabilities_;
    const float scale = 1.0f / static_cast<float>(gradient.N());
    float* gradient_data = gradient.raw_data();
    for (int n = 0; n < gradient.N(); ++n) {
        const int base = gradient.offset_unchecked(n, 0, 0, 0);
        gradient_data[base + cached_targets_.at(static_cast<std::size_t>(n))] -= 1.0f;
        for (int i = 0; i < gradient.W(); ++i) {
            gradient_data[base + i] *= scale;
        }
    }
    return gradient;
}
