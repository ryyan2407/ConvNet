#include <cassert>
#include <filesystem>

#include "model_config.hpp"
#include "project_config.hpp"

int main() {
    const std::filesystem::path config_path =
        std::filesystem::path(CNN_CPP_PROJECT_ROOT) / "configs" / "sample_infer_arch.txt";
    Sequential model = build_model_from_config(config_path.string());

    Tensor input(1, 1, 8, 8);
    for (int i = 0; i < input.size(); ++i) {
        input.raw_data()[i] = static_cast<float>(i % 7) / 7.0f;
    }

    Tensor output = model.predict(input);
    assert(output.N() == 1);
    assert(output.C() == 1);
    assert(output.H() == 1);
    assert(output.W() == 3);
    return 0;
}
