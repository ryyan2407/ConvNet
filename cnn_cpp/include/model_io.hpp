#pragma once

#include <string>

#include "sequential.hpp"

void save_model_artifact(const Sequential& model,
                         const std::string& directory,
                         const std::string& manifest_filename = "model.txt");

Sequential load_model_artifact(const std::string& manifest_path);
