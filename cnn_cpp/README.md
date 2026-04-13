# `cnn_cpp`

From-scratch convolutional neural network engine in modern C++.

This project started as a Stage 1 inference engine and then grew into a small Stage 2 training stack. It does not use PyTorch, TensorFlow, OpenCV, Eigen, Armadillo, or other ML/math libraries for the core engine. The CNN layers, tensor storage, forward pass, backward pass, dataset loading, and training loop are implemented in this repo.

The project now also supports:

- architecture config files
- saved model artifacts
- checkpoint resume
- standalone evaluation

## What This Project Includes

- 4D tensor container with layout `[N, C, H, W]`
- Forward and backward support for:
  - `Conv2D`
  - `ReLU`
  - `MaxPool2D`
  - `Flatten`
  - `Linear`
- Inference-only `Softmax`
- `Sequential` model container
- `CrossEntropyLoss`
- `SGD` optimizer
- architecture config loading
- model artifact save/load
- checkpoint metadata
- standalone evaluation executable
- Plain-text weight loading and saving
- Plain-text image / dataset loading
- Native MNIST IDX parsing
- Small demos, tests, and a benchmark executable

## Current Scope

### Stage 1

Stage 1 is the inference engine:

- tensor storage
- layer execution
- sequential model composition
- sample input loading
- plain-text weight loading
- simple inference executable

### Stage 2

Stage 2 adds training support:

- cached forward state where needed
- `backward()` for trainable and non-trainable layers
- parameter gradients
- SGD updates
- cross-entropy loss over logits
- dataset ingestion for training data
- checkpoint save/load
- evaluation on saved artifacts
- resume training from saved checkpoints

## Real Result

This repo is beyond the "wiring only" stage. A real MNIST run with the current code reached:

- `10,000` training samples
- `10` epochs
- final test accuracy: `0.9464`

## Project Layout

```text
cnn_cpp/
├── CMakeLists.txt
├── README.md
├── data/
├── include/
├── src/
├── tests/
└── weights/
```

Important entry points:

- [src/main.cpp](/home/ryyan/convnet/cnn_cpp/src/main.cpp:1)
  Inference example using sample input and plain-text weights.
- [src/first_milestone.cpp](/home/ryyan/convnet/cnn_cpp/src/first_milestone.cpp:1)
  Minimal `4x4 -> Conv2D -> ReLU` sanity demo.
- [src/train_demo.cpp](/home/ryyan/convnet/cnn_cpp/src/train_demo.cpp:1)
  Small synthetic training demo with save/load roundtrip.
- [src/mnist_train.cpp](/home/ryyan/convnet/cnn_cpp/src/mnist_train.cpp:1)
  Real IDX-based training executable.
- [src/eval.cpp](/home/ryyan/convnet/cnn_cpp/src/eval.cpp:1)
  Dedicated evaluation executable for saved model artifacts.
- [src/benchmark_mnist.cpp](/home/ryyan/convnet/cnn_cpp/src/benchmark_mnist.cpp:1)
  Forward / training-step benchmark with repeated runs.

Core headers:

- [include/tensor.hpp](/home/ryyan/convnet/cnn_cpp/include/tensor.hpp:1)
- [include/layer.hpp](/home/ryyan/convnet/cnn_cpp/include/layer.hpp:1)
- [include/conv2d.hpp](/home/ryyan/convnet/cnn_cpp/include/conv2d.hpp:1)
- [include/maxpool2d.hpp](/home/ryyan/convnet/cnn_cpp/include/maxpool2d.hpp:1)
- [include/linear.hpp](/home/ryyan/convnet/cnn_cpp/include/linear.hpp:1)
- [include/sequential.hpp](/home/ryyan/convnet/cnn_cpp/include/sequential.hpp:1)
- [include/cross_entropy.hpp](/home/ryyan/convnet/cnn_cpp/include/cross_entropy.hpp:1)
- [include/sgd.hpp](/home/ryyan/convnet/cnn_cpp/include/sgd.hpp:1)

## Architecture Notes

### Tensor Representation

The engine keeps a single 4D tensor shape throughout the codebase:

- `N`: batch
- `C`: channels
- `H`: height
- `W`: width

Even flattened vectors are represented as `[N, 1, 1, features]`. That keeps the layer interfaces uniform and avoids introducing a second tensor type.

### Training vs Inference

Layers expose both:

- training forward paths that may cache data for backward
- inference paths that avoid stomping those caches

This matters for correctness and also allowed safe parallel evaluation work without corrupting training state.

### Convolution Implementation

`Conv2D` no longer uses the original direct nested loops. It was replaced with an `im2col`-style path and then optimized further:

- `im2col` / `col2im` helpers
- faster GEMM-like forward accumulation
- optimized backward input accumulation
- optimized weight-gradient accumulation using packed / transposed column storage

That is still all implemented locally in C++ in [src/conv2d.cpp](/home/ryyan/convnet/cnn_cpp/src/conv2d.cpp:1).

## Build

```bash
mkdir -p build
cd build
cmake ../cnn_cpp
cmake --build .
```

Run tests:

```bash
ctest --output-on-failure
```

### Build Defaults

- If no build type is provided, CMake defaults to `Release`
- Release builds use `-O3`
- `CNN_CPP_ENABLE_NATIVE_OPT=ON` adds `-march=native` for GCC / Clang in Release builds
- OpenMP is enabled automatically if the compiler supports it

Example explicit release build:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../cnn_cpp
cmake --build .
```

## Executables

### `cnn_infer`

Runs the sample inference pipeline from [src/main.cpp](/home/ryyan/convnet/cnn_cpp/src/main.cpp:1).

Current example architecture:

- input `1x1x8x8`
- `Conv2D(1 -> 2)`
- `ReLU`
- `MaxPool2D`
- `Conv2D(2 -> 3)`
- `ReLU`
- `MaxPool2D`
- `Flatten`
- `Linear`
- `Softmax`

`cnn_infer` can load either:

- a saved model artifact manifest, or
- an architecture config file

Run:

```bash
./cnn_infer
./cnn_infer /home/ryyan/convnet/cnn_cpp/weights/sample_infer_model.txt
./cnn_infer /home/ryyan/convnet/cnn_cpp/configs/sample_infer_arch.txt
```

### `cnn_first_milestone`

Smallest useful forward-pass demo:

- `4x4` input
- one `Conv2D`
- one `ReLU`

Run:

```bash
./cnn_first_milestone
```

### `cnn_train_demo`

Synthetic training demo for verifying:

- forward
- backward
- optimizer updates
- checkpoint save/load

Run:

```bash
./cnn_train_demo
```

### `cnn_mnist_train`

Training executable for real IDX datasets.

Usage:

```bash
./cnn_mnist_train <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [max_samples] [epochs] [batch_size] [test-images.idx3-ubyte] [test-labels.idx1-ubyte] [checkpoint_dir] [model_config] [resume_artifact_dir]
```

Example:

```bash
./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 1000 3 32
```

Real training example:

```bash
./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 10000 10 32 /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte full_mnist_run_10k_10e
```

Behavior:

- expects `28x28` grayscale IDX images
- auto-detects `t10k-images.idx3-ubyte` and `t10k-labels.idx1-ubyte` in the same folder if test paths are omitted
- builds the model from a config file
- saves checkpoints under `best/`, `final/`, and per-improvement epoch folders
- writes `training_state.txt` alongside saved artifacts
- can resume from a saved artifact directory

Important note:

- tiny runs such as `64 samples, 1 epoch` are only lifecycle smoke tests
- they are useful for validating `train -> save -> eval -> resume`
- they are not representative model-quality runs

### `cnn_eval`

Dedicated evaluation executable for saved model artifacts.

Usage:

```bash
./cnn_eval <model_artifact_manifest> <images.idx3-ubyte> <labels.idx1-ubyte> [max_samples] [batch_size]
```

Example:

```bash
./cnn_eval /home/ryyan/convnet/build/full_mnist_run_10k_10e/final/model.txt /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte 10000 32
```

### `cnn_benchmark`

Performance benchmark for inference and training-step throughput.

Usage:

```bash
./cnn_benchmark <train-images.idx3-ubyte> <train-labels.idx1-ubyte> [samples] [batch_size] [iterations] [repeats] [warmup_repeats] [threads]
```

Defaults:

- `samples=512`
- `batch_size=32`
- `iterations=50`
- `repeats=3`
- `warmup_repeats=1`
- `threads=0` meaning "use current OpenMP default"

Recommended comparison command:

```bash
./cnn_benchmark /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 512 32 50 3 1 1
```

Why this is the recommended command:

- fixed sample count
- fixed batch size
- longer runs
- repeated measurements
- explicit warmup
- single-thread execution for lower variance during kernel work

The benchmark now prints:

- per-repeat total times
- median and mean throughput
- layer breakdown from a median-like representative run
- internal `Conv2D` timing breakdowns

## Data and Weight Formats

### Architecture Configs

Architecture configs are plain text files with header:

```text
cnn_cpp_config_v1
```

Included examples:

- [configs/mnist_cnn.txt](/home/ryyan/convnet/cnn_cpp/configs/mnist_cnn.txt)
- [configs/sample_infer_arch.txt](/home/ryyan/convnet/cnn_cpp/configs/sample_infer_arch.txt)

These define architecture only, not learned weights.

### Model Artifacts

Saved model artifacts are plain text manifest directories with header:

```text
cnn_cpp_model_v1
```

An artifact directory contains:

- `model.txt`
- one weight file per trainable layer
- one bias file per trainable layer
- optionally `training_state.txt` when written by the trainer

Config vs artifact:

- config = architecture only
- artifact = architecture + weights

### Plain-Text Image Input

`load_image_as_tensor(...)` can load text files containing pixel values and normalize them into a tensor.

Example sample input:

- [data/sample_input.txt](/home/ryyan/convnet/cnn_cpp/data/sample_input.txt)

### Plain-Text Dataset Format

Text dataset format:

- `images.txt`
  one flattened sample per line
- `labels.txt`
  one integer label per line

Included demo files:

- [data/demo_train_images.txt](/home/ryyan/convnet/cnn_cpp/data/demo_train_images.txt)
- [data/demo_train_labels.txt](/home/ryyan/convnet/cnn_cpp/data/demo_train_labels.txt)

### Native IDX Support

Supported in [src/image_loader.cpp](/home/ryyan/convnet/cnn_cpp/src/image_loader.cpp:1).

Expected MNIST magic numbers:

- images: `2051`
- labels: `2049`

Loaded shape:

- image samples become `[N, 1, H, W]`

### Weight Files

Weights are plain text floats.

Example files:

- [weights/conv1_weights.txt](/home/ryyan/convnet/cnn_cpp/weights/conv1_weights.txt)
- [weights/conv1_bias.txt](/home/ryyan/convnet/cnn_cpp/weights/conv1_bias.txt)
- [weights/conv2_weights.txt](/home/ryyan/convnet/cnn_cpp/weights/conv2_weights.txt)
- [weights/conv2_bias.txt](/home/ryyan/convnet/cnn_cpp/weights/conv2_bias.txt)
- [weights/fc_weights.txt](/home/ryyan/convnet/cnn_cpp/weights/fc_weights.txt)
- [weights/fc_bias.txt](/home/ryyan/convnet/cnn_cpp/weights/fc_bias.txt)
- [weights/sample_infer_model.txt](/home/ryyan/convnet/cnn_cpp/weights/sample_infer_model.txt)

Trained-demo checkpoint examples:

- [weights/trained_demo/conv_weights.txt](/home/ryyan/convnet/cnn_cpp/weights/trained_demo/conv_weights.txt)
- [weights/trained_demo/conv_bias.txt](/home/ryyan/convnet/cnn_cpp/weights/trained_demo/conv_bias.txt)
- [weights/trained_demo/linear_weights.txt](/home/ryyan/convnet/cnn_cpp/weights/trained_demo/linear_weights.txt)
- [weights/trained_demo/linear_bias.txt](/home/ryyan/convnet/cnn_cpp/weights/trained_demo/linear_bias.txt)

## Tests

Current tests cover:

- tensor indexing and storage
- ReLU behavior
- Conv2D correctness
- MaxPool2D correctness
- Flatten shape and order
- Linear correctness
- Softmax correctness
- sequential execution
- file I/O and IDX parsing
- Stage 2 training flow
- numerical gradient checking
- model artifact roundtrip
- model config construction
- checkpoint state roundtrip

Test files live under [tests](/home/ryyan/convnet/cnn_cpp/tests).

Run all tests:

```bash
ctest --output-on-failure
```

The current suite passes `14/14`.

## Performance Summary

The code has gone through several performance passes:

- cache-friendly tensor access
- zero-copy batch views
- label views instead of per-batch label-vector allocation
- OpenMP on safe forward / evaluation paths
- separate cache-free inference path
- `im2col`-based convolution
- optimized `Conv2D` forward and backward accumulation
- specialized `MaxPool2D` fast path for `2x2, stride=2`
- repeated-run benchmark with median / mean reporting

This project is now well past the initial "proof of concept" phase. It is still a learning / engineering project, but it is no longer just a pile of random `.cpp` files.

## End-To-End Workflow

The project now supports a complete small-engine workflow:

1. Build a model from config.
2. Train it on IDX data with `cnn_mnist_train`.
3. Save checkpoints as model artifacts.
4. Evaluate artifacts with `cnn_eval`.
5. Resume training from a saved artifact directory.

Lifecycle smoke test:

```bash
./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 64 1 32 /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte stage2_lifecycle_ckpt
./cnn_eval /home/ryyan/convnet/build/stage2_lifecycle_ckpt/final/model.txt /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte 64 32
./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 64 1 32 /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte stage2_resume_ckpt /home/ryyan/convnet/cnn_cpp/configs/mnist_cnn.txt /home/ryyan/convnet/build/stage2_lifecycle_ckpt/final
```

Representative real training run:

```bash
./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 10000 10 32 /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte full_mnist_run_10k_10e
```

## Limitations

This is still a small educational / experimental engine, not a production deep learning framework.

Current limitations:

- no GPU support
- no automatic mixed precision
- no external BLAS backend
- no advanced optimizer set beyond SGD
- no convolutions beyond the current small-kernel implementation
- no serialization format beyond plain text weights
- limited model architecture support compared with full ML frameworks
- benchmarking can still vary across machines, thread counts, and thermal states

## If You Want To Extend It

Natural next steps:

- improve `Conv2D` matmul blocking further
- add more optimizers
- add loss / metric variants
- support saving complete model manifests
- add dataset split helpers
- expand beyond MNIST-style grayscale inputs
- add more rigorous benchmark automation

## License

This repository includes an MIT license at the repo root.
