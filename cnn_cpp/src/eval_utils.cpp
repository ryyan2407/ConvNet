#include "eval_utils.hpp"

#include <algorithm>
#include <cstdint>

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
