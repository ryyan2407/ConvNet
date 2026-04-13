#include <cassert>
#include <cmath>
#include <filesystem>
#include <memory>

#include "conv2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "maxpool2d.hpp"
#include "model_io.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "softmax.hpp"

namespace {

float max_abs_diff(const Tensor& a, const Tensor& b) {
    assert(a.size() == b.size());
    float max_diff = 0.0f;
    for (int i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a.raw_data()[i] - b.raw_data()[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

}  // namespace

int main() {
    Sequential model;

    auto conv = std::make_unique<Conv2D>(1, 2, 3, 1, 1);
    conv->set_weights({
        0.10f, 0.00f, -0.10f,
        0.05f, 0.20f, -0.05f,
        0.02f, 0.03f, -0.01f,
        -0.07f, 0.04f, 0.08f,
        0.09f, -0.02f, 0.01f,
        0.03f, -0.06f, 0.05f
    });
    conv->set_bias({0.01f, -0.02f});
    model.add(std::move(conv));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2));
    model.add(std::make_unique<Flatten>());

    auto linear = std::make_unique<Linear>(8, 3);
    linear->set_weights({
        0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.70f, -0.80f,
        -0.15f, 0.25f, -0.35f, 0.45f, -0.55f, 0.65f, -0.75f, 0.85f,
        0.05f, 0.04f, 0.03f, 0.02f, 0.01f, -0.01f, -0.02f, -0.03f
    });
    linear->set_bias({0.11f, -0.12f, 0.13f});
    model.add(std::move(linear));
    model.add(std::make_unique<Softmax>());

    Tensor input(1, 1, 4, 4);
    const float values[16] = {
        0.0f, 0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 0.1f,
        0.2f, 0.3f, 0.4f, 0.5f
    };
    for (int i = 0; i < 16; ++i) {
        input.raw_data()[i] = values[i];
    }

    const Tensor before = model.predict(input);

    const std::filesystem::path artifact_dir = std::filesystem::path("model_io_roundtrip_artifact");
    save_model_artifact(model, artifact_dir.string());
    Sequential reloaded = load_model_artifact((artifact_dir / "model.txt").string());
    const Tensor after = reloaded.predict(input);

    assert(max_abs_diff(before, after) < 1e-6f);
    return 0;
}
