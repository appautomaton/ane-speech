"""Qwen3-ASR, its Confucius4-R2T2 finetune, and Qwen3-ForcedAligner: MLX traces
to Core AI models.

All three checkpoints share one architecture, so one converter serves them. The
two recognizers share their audio encoder bit for bit (R2T2 finetuned only the
decoder); the aligner is a smaller model whose decoder head predicts time
classes instead of tokens, and it keeps its own encoder. Weights are read in
place at conversion; the runtime reads only artifacts/qwen3_asr/:

  audio_encoder/   the recognizers' shared encoder; tracing one checks that its
                   encoder matches what is there
  <checkpoint>/    its decoder, and runtime/: prompt and front-end constants,
                   the embedding table, and the tokenizer; the aligner's also
                   holds its own audio_encoder/

Scripts pick a checkpoint with `--checkpoint`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert.workspace import ARTIFACTS, mlx_models_dir, weights_dir

CHECKPOINTS = {
    "confucius4_r2t2": lambda: mlx_models_dir() / "netease/confucius4_r2t2/mlx-bf16",
    "qwen3_asr_1_7b": lambda: weights_dir() / "speech/ASR/Qwen3-ASR-1.7B",  # the official bf16 release
    "qwen3_forced_aligner_0_6b": lambda: mlx_models_dir() / "qwen3_forced_aligner_0_6b/original",  # upstream bf16
}
ALIGNERS = {"qwen3_forced_aligner_0_6b"}
DEFAULT = "confucius4_r2t2"
ENCODER = "audio_encoder"


def sample_wav() -> Path:
    """A short Chinese clip that ships with the R2T2 checkpoint; any checkpoint can read it."""
    return mlx_models_dir() / "netease/confucius4_r2t2/samples/test.wav"


def align_clips() -> list[tuple[Path, str, str]]:
    """Clips with their exact transcripts and language, for the aligner's references."""
    english = mlx_models_dir() / "ibm/granite_4_0_1b_speech/mlx-int8/multilingual_sample.wav"
    return [
        (sample_wav(), "之前有顾客自己带酒水，也没加收钱或者不让喝。", "zh"),
        (english, "For Timothy was a spoiled cat, and he allowed no one to interfere. Everybody waited "
         "upon him, moving their chairs even, for he was monarch of the hearth. Dinarzad, the night "
         "following, called his sister when it was time. \"If you do not sleep, my sister,\" she said "
         "to him, \"I pray you, in waiting the day which will soon appear, to continue the tale of the "
         "fisherman.\"", "en"),
    ]


def add_checkpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=DEFAULT,
                        help=f"weights to read (default {DEFAULT}); output goes to artifacts/qwen3_asr/<checkpoint>/")


def model_dir(checkpoint: str) -> Path:
    return CHECKPOINTS[checkpoint]()


def artifacts_dir(checkpoint: str) -> Path:
    return ARTIFACTS / "qwen3_asr" / checkpoint


def encoder_dir(checkpoint: str) -> Path:
    """The recognizers share one encoder; the aligner keeps its own."""
    if checkpoint in ALIGNERS:
        return artifacts_dir(checkpoint) / ENCODER
    return ARTIFACTS / "qwen3_asr" / ENCODER
