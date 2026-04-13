#!/usr/bin/env python3

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path("/home/ryyan/convnet")
SCREENSHOT_DIR = ROOT / "docs" / "screenshots"
RELEASE_DIR = ROOT / "docs" / "releases"

FONT_SIZE = 16
LINE_HEIGHT = 24
PADDING_X = 28
PADDING_Y = 28
CHAR_WIDTH = 9


def render_terminal_svg(title: str, command: str, body: str, output_path: Path) -> None:
    lines = [f"$ {command}"] + body.rstrip("\n").splitlines()
    max_len = max(len(line) for line in lines) if lines else 0
    width = PADDING_X * 2 + max_len * CHAR_WIDTH
    header_h = 42
    body_h = PADDING_Y * 2 + len(lines) * LINE_HEIGHT
    height = header_h + body_h

    y = header_h + PADDING_Y
    text_chunks = []
    for i, line in enumerate(lines):
        fill = "#8be9fd" if i == 0 else "#f8f8f2"
        text_chunks.append(
            f'<text x="{PADDING_X}" y="{y + i * LINE_HEIGHT}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace" '
            f'font-size="{FONT_SIZE}" fill="{fill}">{html.escape(line)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="16" fill="#0b1020"/>
  <rect x="0" y="0" width="{width}" height="{header_h}" rx="16" fill="#151b2f"/>
  <rect x="0" y="{header_h}" width="{width}" height="{height - header_h}" fill="#0b1020"/>
  <circle cx="22" cy="21" r="6" fill="#ff5f56"/>
  <circle cx="42" cy="21" r="6" fill="#ffbd2e"/>
  <circle cx="62" cy="21" r="6" fill="#27c93f"/>
  <text x="84" y="26"
        font-family="Inter, Segoe UI, Arial, sans-serif"
        font-size="15" fill="#c9d1d9">{html.escape(title)}</text>
  {''.join(text_chunks)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    infer_body = """Input shape: [1, 1, 8, 8]
Model file: "/home/ryyan/convnet/cnn_cpp/weights/sample_infer_model.txt"
Output shape: [1, 1, 1, 3]
Probabilities:
class 0: 0.360481
class 1: 0.307937
class 2: 0.331582
Predicted class: 0
"""

    eval_body = """Evaluated model: /home/ryyan/convnet/build/full_mnist_run_10k_10e/final/model.txt
Samples: 10000
Loss: 0.181209
Accuracy: 0.9464
"""

    train_body = """epoch 1 loss=2.28339 accuracy=0.1938 test_loss=2.23702 test_accuracy=0.3791
epoch 2 loss=1.46234 accuracy=0.6275 test_loss=0.581128 test_accuracy=0.8383
epoch 3 loss=0.472504 accuracy=0.8567 test_loss=0.365774 test_accuracy=0.894
epoch 4 loss=0.361759 accuracy=0.8892 test_loss=0.316316 test_accuracy=0.9096
epoch 5 loss=0.306698 accuracy=0.9085 test_loss=0.292099 test_accuracy=0.9145
epoch 6 loss=0.272997 accuracy=0.918 test_loss=0.248705 test_accuracy=0.926
epoch 7 loss=0.244988 accuracy=0.9259 test_loss=0.230148 test_accuracy=0.9329
epoch 8 loss=0.218691 accuracy=0.9347 test_loss=0.218912 test_accuracy=0.9326
epoch 9 loss=0.199534 accuracy=0.9432 test_loss=0.204416 test_accuracy=0.9394
epoch 10 loss=0.182134 accuracy=0.9477 test_loss=0.181209 test_accuracy=0.9464
sample0 label=5 pred=5 p=0.830617
test_sample0 label=7 pred=7 p=0.999541
Saved final weights to: "full_mnist_run_10k_10e/final"
Saved best checkpoint to: "full_mnist_run_10k_10e/best"
"""

    render_terminal_svg(
        "Inference",
        "./cnn_infer /home/ryyan/convnet/cnn_cpp/weights/sample_infer_model.txt",
        infer_body,
        SCREENSHOT_DIR / "cnn_infer.svg",
    )
    render_terminal_svg(
        "Training",
        "./cnn_mnist_train /home/ryyan/convnet/images/train-images.idx3-ubyte /home/ryyan/convnet/images/train-labels.idx1-ubyte 10000 10 32 /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte full_mnist_run_10k_10e",
        train_body,
        SCREENSHOT_DIR / "cnn_mnist_train.svg",
    )
    render_terminal_svg(
        "Evaluation",
        "./cnn_eval /home/ryyan/convnet/build/full_mnist_run_10k_10e/final/model.txt /home/ryyan/convnet/images/t10k-images.idx3-ubyte /home/ryyan/convnet/images/t10k-labels.idx1-ubyte 10000 32",
        eval_body,
        SCREENSHOT_DIR / "cnn_eval.svg",
    )


if __name__ == "__main__":
    main()
