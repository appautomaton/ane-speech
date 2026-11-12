"""Activation ranges of the Qwen3-ASR audio encoder in MLX, against fp16's limit.

The ANE computes in fp16, whose largest finite value is 65504. This script runs
mlx-speech's audio encoder on one clip and records the largest magnitude each
probed tensor reaches, plus each LayerNorm input's per-token sum of (x - mean)^2,
which a LayerNorm lowered to an fp16 sum must hold. Run it from the repo root
under mlx-speech's environment; MLX needs Metal, so run it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_encoder_ranges

`--checkpoint` picks the weights (default Confucius4-R2T2); the default clip is
R2T2's Chinese sample. The default
`--dtype bf16` feeds bf16 features, as mlx-speech's fast streaming compute does;
`--dtype f32` promotes the activations to float32, as its reference compute does.

The probed forward mirrors `Qwen3ASRAudioEncoder.__call__` and must reproduce the
encoder's own output exactly; the script fails otherwise.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_speech.models.qwen3_asr.audio_encoder import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRAudioEncoderLayer,
    _activation,
    _attention_block_lengths,
    _iter_blocks,
    _split_first_axis,
)
from mlx_speech.models.qwen3_asr.checkpoint import load_qwen3_asr_checkpoint
from mlx_speech.models.qwen3_asr.feature_extraction import Qwen3ASRFeatureExtractor

from convert.qwen3_asr import add_checkpoint_argument, model_dir, sample_wav

FP16_MAX = 65504.0
NEAR = FP16_MAX / 10

DTYPES = {"bf16": mx.bfloat16, "f32": mx.float32}
STAGES = {
    "conv2d1": "conv2d1, pre-GELU",
    "conv2d2": "conv2d2, pre-GELU",
    "conv2d3": "conv2d3, pre-GELU",
    "conv_out": "conv_out",
    "embed": "conv_out + positions",
    "ln_post.sumsq": "ln_post input, sum (x-mean)^2",
    "ln_post": "ln_post",
    "proj1": "proj1, pre-GELU",
    "proj2": "proj2 (encoder output)",
}


class Probe:
    """Running max |x| per name, kept lazy until `values()`."""

    def __init__(self) -> None:
        self._maxima: dict[str, list[mx.array]] = defaultdict(list)

    def __call__(self, name: str, x: mx.array) -> mx.array:
        self._maxima[name].append(mx.max(mx.abs(x)).astype(mx.float32))
        return x

    def norm_input(self, name: str, x: mx.array) -> None:
        """Per-token sum of (x - mean)^2 over channels, the largest value a
        LayerNorm holds if it is lowered to an fp16 sum before dividing."""
        x = x.astype(mx.float32)
        self(name, mx.sum(mx.square(x - mx.mean(x, axis=-1, keepdims=True)), axis=-1))

    def values(self) -> dict[str, float]:
        mx.eval(self._maxima)
        return {name: max(m.item() for m in ms) for name, ms in self._maxima.items()}


def load_encoder(model_dir: Path) -> Qwen3ASRAudioEncoder:
    """The audio tower alone, loaded and cast as mlx-speech's runtime loads it."""
    checkpoint = load_qwen3_asr_checkpoint(model_dir)
    encoder = Qwen3ASRAudioEncoder(checkpoint.config.audio_config)
    prefix = "audio_tower."
    weights = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.state_dict.items()
        if key.startswith(prefix)
    }
    # Generated at construction, not stored; supplying it keeps the load strict.
    weights["positional_embedding.positional_embedding"] = (
        encoder.positional_embedding.positional_embedding
    )
    encoder.load_weights(list(weights.items()), strict=True)
    encoder.set_dtype(mx.bfloat16)
    encoder.eval()
    return encoder


def probed_layer(
    layer: Qwen3ASRAudioEncoderLayer,
    x: mx.array,
    blocks: list[tuple[int, int]],
    probe: Probe,
    i: int,
) -> mx.array:
    attn = layer.self_attn
    heads, head_dim = attn.num_heads, attn.head_dim

    probe.norm_input(f"{i}.ln_sumsq", x)
    normed = layer.self_attn_layer_norm(x)
    outputs = []
    for start, end in blocks:
        block, n = normed[start:end], end - start
        q, k, v = (
            probe(f"{i}.{name}", proj(block)).reshape(n, heads, head_dim).transpose(1, 0, 2)
            for name, proj in (("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj))
        )
        scores = probe(f"{i}.logits", (q @ k.transpose(0, 2, 1)) * attn.scale)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (weights @ v).transpose(1, 0, 2).reshape(n, attn.embed_dim)
        outputs.append(probe(f"{i}.attn_out", attn.out_proj(out)))
    x = probe(f"{i}.resid", x + mx.concatenate(outputs, axis=0))

    probe.norm_input(f"{i}.ln_sumsq", x)
    hidden = probe(f"{i}.ffn_hidden", layer.fc1(layer.final_layer_norm(x)))
    out = probe(f"{i}.ffn_out", layer.fc2(_activation(hidden, layer.activation_function)))
    return probe(f"{i}.resid", x + out)


def probed_forward(
    encoder: Qwen3ASRAudioEncoder, features: mx.array, probe: Probe
) -> tuple[mx.array, list[tuple[int, int]]]:
    """`Qwen3ASRAudioEncoder.__call__` for one clip, with every stage probed."""
    chunked = encoder._chunk_features(features, features.shape[1])

    # Padded frames of the last chunk are probed too: a fixed-shape graph computes them.
    parts = []
    for part in _split_first_axis(chunked.padded_features[:, :, :, None], encoder.conv_chunksize):
        for name in ("conv2d1", "conv2d2", "conv2d3"):
            part = nn.gelu(probe(name, getattr(encoder, name)(part)))
        parts.append(part)
    x = mx.concatenate(parts, axis=0)
    batch, freq, time, channels = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(batch, time, channels * freq)
    x = probe("conv_out", encoder.conv_out(x))
    x = probe("embed", x + encoder.positional_embedding(time)[None].astype(x.dtype))
    x = mx.concatenate([x[c, :n] for c, n in enumerate(chunked.aftercnn_lengths) if n > 0], axis=0)

    lengths = _attention_block_lengths(
        total_length=x.shape[0],
        max_chunk_aftercnn=max(chunked.aftercnn_lengths),
        n_window=encoder.n_window,
        n_window_infer=encoder.n_window_infer,
    )
    blocks = list(_iter_blocks(lengths))
    for i, layer in enumerate(encoder.layers):
        x = probed_layer(layer, x, blocks, probe, i)

    probe.norm_input("ln_post.sumsq", x)
    x = probe("ln_post", encoder.ln_post(x))
    x = probe("proj1", encoder.proj1(x))
    x = probe("proj2", encoder.proj2(_activation(x, encoder.activation_function)))
    return x, blocks


def fmt(value: float) -> str:
    return f"{value:,.1f}"


def flag(value: float) -> str:
    if value > FP16_MAX:
        return " **over**"
    if value >= NEAR:
        return " *near*"
    return ""


def report(values: dict[str, float], layers: int) -> list[str]:
    lines = [
        "| Layer | Residual | Attn logits | FFN hidden | Largest other | LN input, sum (x-mean)^2 |",
        "| ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    others = ("q", "k", "v", "attn_out", "ffn_out")
    for i in range(layers):
        get = lambda name: values[f"{i}.{name}"]  # noqa: E731
        other = max(others, key=get)
        lines.append(
            f"| {i} | {fmt(get('resid'))}{flag(get('resid'))}"
            f" | {fmt(get('logits'))}{flag(get('logits'))}"
            f" | {fmt(get('ffn_hidden'))}{flag(get('ffn_hidden'))}"
            f" | {other} {fmt(get(other))}{flag(get(other))}"
            f" | {fmt(get('ln_sumsq'))}{flag(get('ln_sumsq'))} |"
        )
    lines += ["", "| Stage | Max abs |", "| --- | ---: |"]
    for name, label in STAGES.items():
        lines.append(f"| {label} | {fmt(values[name])}{flag(values[name])} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_checkpoint_argument(parser)
    parser.add_argument("--wav", type=Path, default=sample_wav())
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    args = parser.parse_args()
    checkpoint = model_dir(args.checkpoint)

    encoder = load_encoder(checkpoint)
    extractor = Qwen3ASRFeatureExtractor.from_dir(checkpoint)
    batch = extractor(args.wav)
    frames = int(batch.feature_attention_mask.sum())
    features = mx.array(batch.input_features[0, :, :frames], dtype=DTYPES[args.dtype])

    probe = Probe()
    output, blocks = probed_forward(encoder, features, probe)
    expected = encoder(features, feature_lens=frames).last_hidden_state
    if output.dtype != expected.dtype or not mx.array_equal(output, expected).item():
        diff = mx.max(mx.abs(output.astype(mx.float32) - expected.astype(mx.float32))).item()
        raise SystemExit(f"Probed forward diverges from the encoder: max abs diff {diff}.")
    values = probe.values()

    tensors = {name: v for name, v in values.items() if not name.endswith("sumsq")}
    sums = {name: v for name, v in values.items() if name.endswith("sumsq")}
    print(f"checkpoint   {checkpoint}")
    print(f"clip         {args.wav} ({frames / 100:.2f} s, {frames} mel frames)")
    print(f"compute      {args.dtype} features -> {output.dtype} activations, mlx {mx.__version__}")
    print(f"tokens       {output.shape[0]} in attention blocks of {[e - s for s, e in blocks]}")
    print("parity       probed forward == encoder output, bit for bit")
    for label, group in (("peak tensor", tensors), ("peak LN sum", sums)):
        name = max(group, key=group.get)
        print(f"{label}  {name} {fmt(group[name])}, {FP16_MAX / group[name]:.1f}x below fp16 max")
    print()
    print("\n".join(report(values, len(encoder.layers))))
    print()
    print(f"*near*: within 10x of fp16 max ({fmt(NEAR)}). **over**: above {fmt(FP16_MAX)}.")


if __name__ == "__main__":
    main()
