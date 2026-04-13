# ConvNet

From-scratch convolutional neural network engine in C++.

This repository contains a small CNN project built without external ML frameworks. The core engine, tensor storage, layers, forward pass, backward pass, dataset loading, training loop, and benchmarking code are implemented locally in C++.

## Repository Layout

```text
ConvNet/
├── cnn_cpp/        # actual project source
├── images/         # local MNIST IDX files, ignored by git
├── build/          # local build directory, ignored by git
├── LICENSE
└── README.md
```

The project itself lives in [cnn_cpp](cnn_cpp/).

## What’s Implemented

- 4D tensor container with layout `[N, C, H, W]`
- CNN layers:
  - `Conv2D`
  - `ReLU`
  - `MaxPool2D`
  - `Flatten`
  - `Linear`
  - `Softmax`
- `Sequential` model container
- Stage 1 inference pipeline
- Stage 2 training with:
  - backward propagation
  - `CrossEntropyLoss`
  - `SGD`
- model config loading
- model artifact save/load
- checkpoint resume
- dedicated evaluation executable
- Plain-text weight save/load
- Plain-text sample datasets
- Native MNIST IDX parsing
- Unit-style tests
- Performance benchmark with repeated runs and median/mean reporting

## Quick Start

```bash
mkdir -p build
cd build
cmake ../cnn_cpp
cmake --build .
ctest --output-on-failure
```

Useful executables:

```bash
./cnn_infer
./cnn_first_milestone
./cnn_train_demo
./cnn_mnist_train <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [max_samples] [epochs] [batch_size] [test-images.idx3-ubyte] [test-labels.idx1-ubyte] [checkpoint_dir] [model_config] [resume_artifact_dir]
./cnn_eval <model_artifact_manifest> <images.idx3-ubyte> <labels.idx1-ubyte> [max_samples] [batch_size]
./cnn_benchmark <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [samples] [batch_size] [iterations] [repeats] [warmup_repeats] [threads]
```

## Current Status

The project now supports the full visible lifecycle:

- build a model from architecture config
- train on IDX data
- save full model artifacts
- evaluate saved artifacts
- resume training from a saved checkpoint

Real MNIST result from the current codebase:

- `10,000` training samples
- `10` epochs
- final test accuracy: `0.9464`

## Documentation

The detailed project README is here:

- [cnn_cpp/README.md](cnn_cpp/README.md)

That document covers:

- architecture
- Stage 1 and Stage 2 scope
- build and run instructions
- dataset formats
- benchmarking workflow
- performance notes
- limitations and extension ideas

## Notes

- `images/` is intentionally gitignored because it contains local dataset files.
- `build/` is intentionally gitignored because it contains generated artifacts.
- The repo now includes an MIT license.
