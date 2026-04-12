#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "conv2d.hpp"
#include "cross_entropy.hpp"
#include "flatten.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "relu.hpp"
#include "sgd.hpp"

#if defined(CNN_CPP_USE_OPENMP)
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct TimerStat {
    std::string name;
    double milliseconds = 0.0;
};

struct BenchmarkRun {
    double forward_ms = 0.0;
    double train_ms = 0.0;
    std::vector<TimerStat> forward_stats;
    std::vector<TimerStat> train_stats;
    Conv2D::ProfileStats conv1_profile;
    Conv2D::ProfileStats conv2_profile;
};

Tensor make_batch_view(const Tensor& data, int start, int count) {
    return data.slice_n(start, count);
}

LabelView make_label_view(const std::vector<int>& labels, int start, int count) {
    return LabelView{labels.data() + start, count};
}

void add_elapsed(TimerStat& stat, Clock::time_point start, Clock::time_point end) {
    stat.milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
}

void print_metric(const std::string& name, double milliseconds, int items) {
    const double items_per_second = items > 0 ? (1000.0 * static_cast<double>(items) / milliseconds) : 0.0;
    std::cout << name << ": " << milliseconds << " ms total, "
              << items_per_second << " items/s\n";
}

void print_breakdown(const std::string& title, const std::vector<TimerStat>& stats, int items) {
    std::cout << title << " breakdown:\n";
    for (const auto& stat : stats) {
        print_metric("  " + stat.name, stat.milliseconds, items);
    }
}

void print_conv_profile(const std::string& name, const Conv2D::ProfileStats& stats, int items) {
    std::cout << name << " internal breakdown:\n";
    print_metric("  forward_im2col", stats.forward_im2col_ms, items);
    print_metric("  forward_gemm", stats.forward_gemm_ms, items);
    print_metric("  backward_im2col", stats.backward_im2col_ms, items);
    print_metric("  backward_grad_accum", stats.backward_grad_accum_ms, items);
    print_metric("  backward_gemm", stats.backward_gemm_ms, items);
    print_metric("  backward_col2im", stats.backward_col2im_ms, items);
}

double mean_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median_value(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Usage: ./cnn_benchmark <train-images.idx3-ubyte> <train-labels.idx1-ubyte> "
                << "[samples] [batch_size] [iterations] [repeats] [warmup_repeats] [threads]\n";
            return 1;
        }

        const std::string images_path = argv[1];
        const std::string labels_path = argv[2];
        const int sample_limit = argc >= 4 ? std::stoi(argv[3]) : 512;
        const int batch_size = argc >= 5 ? std::stoi(argv[4]) : 32;
        const int iterations = argc >= 6 ? std::stoi(argv[5]) : 50;
        const int repeats = argc >= 7 ? std::stoi(argv[6]) : 3;
        const int warmup_repeats = argc >= 8 ? std::stoi(argv[7]) : 1;
        const int requested_threads = argc >= 9 ? std::stoi(argv[8]) : 0;
        if (batch_size <= 0 || iterations <= 0 || sample_limit <= 0 || repeats <= 0 || warmup_repeats < 0 ||
            requested_threads < 0) {
            throw std::runtime_error(
                "samples, batch_size, iterations, repeats must be positive; warmup_repeats and threads must be non-negative");
        }

#if defined(CNN_CPP_USE_OPENMP)
        omp_set_dynamic(0);
        if (requested_threads > 0) {
            omp_set_num_threads(requested_threads);
        }
        const int effective_threads = omp_get_max_threads();
#else
        const int effective_threads = 1;
        (void)requested_threads;
#endif

        const LabeledDataset dataset = load_idx_dataset(images_path, labels_path, true, sample_limit);
        if (dataset.data.H() != 28 || dataset.data.W() != 28) {
            throw std::runtime_error("cnn_benchmark expects 28x28 IDX images");
        }

        const int effective_batch = std::min(batch_size, dataset.data.N());
        Tensor batch = make_batch_view(dataset.data, 0, effective_batch);
        LabelView labels = make_label_view(dataset.labels, 0, effective_batch);
        const int total_items = effective_batch * iterations;

        Conv2D conv1(1, 8, 3, 1, 1);
        ReLU relu1;
        MaxPool2D pool1(2, 2);
        Conv2D conv2(8, 16, 3, 1, 1);
        ReLU relu2;
        MaxPool2D pool2(2, 2);
        Flatten flatten;
        Linear linear(16 * 7 * 7, 10);
        CrossEntropyLoss loss;
        SGD optimizer(0.01f);

        auto run_once = [&]() {
            BenchmarkRun run;
            run.forward_stats = {
                {"conv1"}, {"relu1"}, {"pool1"}, {"conv2"}, {"relu2"}, {"pool2"}, {"flatten"}, {"linear"}
            };
            run.train_stats = {
                {"conv1_fwd"}, {"relu1_fwd"}, {"pool1_fwd"}, {"conv2_fwd"}, {"relu2_fwd"}, {"pool2_fwd"},
                {"flatten_fwd"}, {"linear_fwd"}, {"loss"}, {"linear_bwd"}, {"flatten_bwd"}, {"pool2_bwd"},
                {"relu2_bwd"}, {"conv2_bwd"}, {"pool1_bwd"}, {"relu1_bwd"}, {"conv1_bwd"}, {"update"}
            };

            conv1.reset_profile_stats();
            conv2.reset_profile_stats();

            auto forward_total_start = Clock::now();
            for (int i = 0; i < iterations; ++i) {
                auto t0 = Clock::now();
                Tensor x1 = conv1.infer(batch);
                auto t1 = Clock::now();
                Tensor x2 = relu1.infer(x1);
                auto t2 = Clock::now();
                Tensor x3 = pool1.infer(x2);
                auto t3 = Clock::now();
                Tensor x4 = conv2.infer(x3);
                auto t4 = Clock::now();
                Tensor x5 = relu2.infer(x4);
                auto t5 = Clock::now();
                Tensor x6 = pool2.infer(x5);
                auto t6 = Clock::now();
                Tensor x7 = flatten.infer(x6);
                auto t7 = Clock::now();
                Tensor logits = linear.infer(x7);
                auto t8 = Clock::now();

                add_elapsed(run.forward_stats[0], t0, t1);
                add_elapsed(run.forward_stats[1], t1, t2);
                add_elapsed(run.forward_stats[2], t2, t3);
                add_elapsed(run.forward_stats[3], t3, t4);
                add_elapsed(run.forward_stats[4], t4, t5);
                add_elapsed(run.forward_stats[5], t5, t6);
                add_elapsed(run.forward_stats[6], t6, t7);
                add_elapsed(run.forward_stats[7], t7, t8);
                (void)logits;
            }
            auto forward_total_end = Clock::now();

            auto train_total_start = Clock::now();
            for (int i = 0; i < iterations; ++i) {
                auto t0 = Clock::now();
                conv1.zero_grad();
                conv2.zero_grad();
                linear.zero_grad();
                Tensor x1 = conv1.forward(batch);
                auto t1 = Clock::now();
                Tensor x2 = relu1.forward(x1);
                auto t2 = Clock::now();
                Tensor x3 = pool1.forward(x2);
                auto t3 = Clock::now();
                Tensor x4 = conv2.forward(x3);
                auto t4 = Clock::now();
                Tensor x5 = relu2.forward(x4);
                auto t5 = Clock::now();
                Tensor x6 = pool2.forward(x5);
                auto t6 = Clock::now();
                Tensor x7 = flatten.forward(x6);
                auto t7 = Clock::now();
                Tensor logits = linear.forward(x7);
                auto t8 = Clock::now();
                (void)loss.forward(logits, labels);
                auto t9 = Clock::now();
                Tensor g = loss.backward();
                g = linear.backward(g);
                auto t10 = Clock::now();
                g = flatten.backward(g);
                auto t11 = Clock::now();
                g = pool2.backward(g);
                auto t12 = Clock::now();
                g = relu2.backward(g);
                auto t13 = Clock::now();
                g = conv2.backward(g);
                auto t14 = Clock::now();
                g = pool1.backward(g);
                auto t15 = Clock::now();
                g = relu1.backward(g);
                auto t16 = Clock::now();
                g = conv1.backward(g);
                auto t17 = Clock::now();
                conv1.update(0.01f);
                conv2.update(0.01f);
                linear.update(0.01f);
                auto t18 = Clock::now();

                add_elapsed(run.train_stats[0], t0, t1);
                add_elapsed(run.train_stats[1], t1, t2);
                add_elapsed(run.train_stats[2], t2, t3);
                add_elapsed(run.train_stats[3], t3, t4);
                add_elapsed(run.train_stats[4], t4, t5);
                add_elapsed(run.train_stats[5], t5, t6);
                add_elapsed(run.train_stats[6], t6, t7);
                add_elapsed(run.train_stats[7], t7, t8);
                add_elapsed(run.train_stats[8], t8, t9);
                add_elapsed(run.train_stats[9], t9, t10);
                add_elapsed(run.train_stats[10], t10, t11);
                add_elapsed(run.train_stats[11], t11, t12);
                add_elapsed(run.train_stats[12], t12, t13);
                add_elapsed(run.train_stats[13], t13, t14);
                add_elapsed(run.train_stats[14], t14, t15);
                add_elapsed(run.train_stats[15], t15, t16);
                add_elapsed(run.train_stats[16], t16, t17);
                add_elapsed(run.train_stats[17], t17, t18);
            }
            auto train_total_end = Clock::now();

            run.forward_ms =
                std::chrono::duration<double, std::milli>(forward_total_end - forward_total_start).count();
            run.train_ms =
                std::chrono::duration<double, std::milli>(train_total_end - train_total_start).count();
            run.conv1_profile = conv1.profile_stats();
            run.conv2_profile = conv2.profile_stats();
            return run;
        };

        for (int i = 0; i < warmup_repeats; ++i) {
            (void)run_once();
        }

        std::vector<BenchmarkRun> runs;
        runs.reserve(static_cast<std::size_t>(repeats));
        std::vector<double> forward_runs;
        std::vector<double> train_runs;
        forward_runs.reserve(static_cast<std::size_t>(repeats));
        train_runs.reserve(static_cast<std::size_t>(repeats));
        for (int i = 0; i < repeats; ++i) {
            runs.push_back(run_once());
            forward_runs.push_back(runs.back().forward_ms);
            train_runs.push_back(runs.back().train_ms);
        }

        const double forward_median = median_value(forward_runs);
        const double train_median = median_value(train_runs);
        const double forward_mean = mean_value(forward_runs);
        const double train_mean = mean_value(train_runs);
        int median_index = 0;
        double best_distance = std::abs(runs[0].train_ms - train_median);
        for (int i = 1; i < repeats; ++i) {
            const double distance = std::abs(runs[static_cast<std::size_t>(i)].train_ms - train_median);
            if (distance < best_distance) {
                best_distance = distance;
                median_index = i;
            }
        }
        const BenchmarkRun& representative_run = runs[static_cast<std::size_t>(median_index)];

        std::cout << "Benchmark config: samples=" << dataset.data.N()
                  << " batch_size=" << effective_batch
                  << " iterations=" << iterations
                  << " repeats=" << repeats
                  << " warmup_repeats=" << warmup_repeats
                  << " threads=" << effective_threads << "\n";
        std::cout << "repeat totals (ms):\n";
        for (int i = 0; i < repeats; ++i) {
            std::cout << "  run " << (i + 1)
                      << ": forward=" << runs[static_cast<std::size_t>(i)].forward_ms
                      << " train_step=" << runs[static_cast<std::size_t>(i)].train_ms << "\n";
        }
        print_metric("forward_median", forward_median, total_items);
        print_metric("forward_mean", forward_mean, total_items);
        print_metric("train_step_median", train_median, total_items);
        print_metric("train_step_mean", train_mean, total_items);
        std::cout << "representative breakdown: median-like run " << (median_index + 1) << "\n";
        print_breakdown("forward", representative_run.forward_stats, total_items);
        print_breakdown("train_step", representative_run.train_stats, total_items);
        print_conv_profile("conv1", representative_run.conv1_profile, total_items);
        print_conv_profile("conv2", representative_run.conv2_profile, total_items);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << "\n";
        return 1;
    }
}
