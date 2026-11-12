"""Qwen3-ASR and its Confucius4-R2T2 finetune: MLX traces to Core AI models.

Both checkpoints share one architecture, so one converter and one runtime serve
them, and R2T2 finetuned only the decoder, so they share the audio encoder bit
for bit. Weights are read in place at conversion; the runtime reads only
artifacts/qwen3_asr/:

  audio_encoder/   the shared encoder; tracing a checkpoint checks that its
                   encoder matches what is there
  <checkpoint>/    its decoder, and runtime/: prompt and front-end constants,
                   the embedding table, and the tokenizer

Scripts pick a checkpoint with `--checkpoint`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert.workspace import ARTIFACTS, mlx_models_dir, weights_dir

CHECKPOINTS = {
    "confucius4_r2t2": lambda: mlx_models_dir() / "netease/confucius4_r2t2/mlx-bf16",
    "qwen3_asr_1_7b": lambda: weights_dir() / "speech/ASR/Qwen3-ASR-1.7B",  # the official bf16 release
}
DEFAULT = "confucius4_r2t2"
ENCODER = "audio_encoder"


def sample_wav() -> Path:
    """A short Chinese clip that ships with the R2T2 checkpoint; any checkpoint can read it."""
    return mlx_models_dir() / "netease/confucius4_r2t2/samples/test.wav"


def add_checkpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=DEFAULT,
                        help=f"weights to read (default {DEFAULT}); output goes to artifacts/qwen3_asr/<checkpoint>/")


def model_dir(checkpoint: str) -> Path:
    return CHECKPOINTS[checkpoint]()


def artifacts_dir(checkpoint: str) -> Path:
    return ARTIFACTS / "qwen3_asr" / checkpoint


def encoder_dir() -> Path:
    return ARTIFACTS / "qwen3_asr" / ENCODER
