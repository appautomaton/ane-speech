"""Trace the Qwen3-ASR audio encoder over one 8-second block, for conversion.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_encoder_trace

The encoder attends within 8-second blocks (n_window_infer mel frames, 104
tokens), so one block is the unit the ANE model computes. This traces
mlx-speech's own modules over a full block and writes to the checkpoint's
encoder folder: the recognizers share artifacts/qwen3_asr/audio_encoder/, and
the aligner keeps its own. When an encoder is already there, the checkpoint's
must match it constant for constant; then nothing is written, and the script
fails if it differs. It writes:

  encoder_block.json       traced primitives in execution order, and the dtype
                           of every constant
  encoder_block.npz        the constants (weights); bf16 is stored as raw uint16
                           bits, so nothing is rounded here
  encoder_reference.npz    MLX outputs for parity checks, in float32: a full block
                           (the first 8 s of a long clip) and a partial one
                           (test.wav: 674 frames, 88 tokens), each computed with
                           bf16 and with float32 activations

The traced function must reproduce the encoder's own output exactly on a full
block; the script fails otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_speech.models.qwen3_asr.audio_encoder import Qwen3ASRAudioEncoder, _activation
from mlx_speech.models.qwen3_asr.feature_extraction import Qwen3ASRFeatureExtractor

from convert.qwen3_asr import add_checkpoint_argument, encoder_dir, model_dir, sample_wav
from convert.workspace import mlx_models_dir
from reference.qwen3_asr_encoder_ranges import load_encoder

LONG_CLIP = Path("examples/audio/vibevoice_4speaker_convo.wav")  # in mlx-speech


def block_function(encoder: Qwen3ASRAudioEncoder):
    """The encoder over exactly one full block, built from mlx-speech's modules."""
    chunk = encoder.n_window * 2
    chunks = encoder.n_window_infer // chunk
    tokens_per_chunk = int(encoder.output_lengths(chunk))
    tokens = chunks * tokens_per_chunk

    def block(mel: mx.array) -> mx.array:
        # The chunking of `_chunk_features`, for a block with no partial chunk.
        x = mel.reshape(mel.shape[0], chunks, chunk).transpose(1, 0, 2)
        x = encoder._conv_forward(x, [tokens_per_chunk] * chunks)
        for layer in encoder.layers:
            x = layer(x, block_lengths=[tokens])
        x = encoder.proj1(encoder.ln_post(x))
        return encoder.proj2(_activation(x, encoder.activation_function))

    return block, tokens


def features(extractor: Qwen3ASRFeatureExtractor, wav: Path) -> np.ndarray:
    batch = extractor(wav)
    return batch.input_features[0, :, : int(batch.feature_attention_mask.sum())]


def run(encoder: Qwen3ASRAudioEncoder, mel: np.ndarray, dtype: mx.Dtype) -> np.ndarray:
    out = encoder(mx.array(mel, dtype=dtype)).last_hidden_state
    return np.array(out.astype(mx.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_checkpoint_argument(parser)
    parser.add_argument("--wav", type=Path, default=sample_wav())
    parser.add_argument("--long-wav", type=Path, default=mlx_models_dir().parent / LONG_CLIP)
    args = parser.parse_args()
    checkpoint, out_dir = model_dir(args.checkpoint), encoder_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(checkpoint)
    mx.eval(encoder.parameters())
    extractor = Qwen3ASRFeatureExtractor.from_dir(checkpoint)
    block, tokens = block_function(encoder)
    frames = encoder.n_window_infer

    full = features(extractor, args.long_wav)[:, :frames]
    partial = features(extractor, args.wav)
    if full.shape[1] != frames or partial.shape[1] >= frames:
        raise SystemExit(f"Need a clip of at least {frames} frames and one shorter than that.")

    mel = mx.array(full, dtype=mx.bfloat16)
    expected = encoder(mel).last_hidden_state
    if not mx.array_equal(block(mel), expected).item():
        raise SystemExit("The block function diverges from the encoder on a full block.")

    records: list[dict] = []
    mx.export_function(records.append, block, mel)
    constants, dtypes, graph = {}, {}, []
    for record in records:
        if record["type"] != "constants":
            graph.append(record)
            continue
        for name, value in record["constants"]:
            dtypes[name] = str(value.dtype).removeprefix("mlx.core.")
            raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
            constants[name] = np.array(raw)
    shared = out_dir / "encoder_block.npz"
    if shared.exists():
        stored = np.load(shared)
        same = set(stored.files) == set(constants) and all(np.array_equal(stored[k], v) for k, v in constants.items())
        if not same:
            raise SystemExit(f"{args.checkpoint}'s audio encoder differs from the one in {out_dir}.")
        print(f"{args.checkpoint}: audio encoder identical to the one in {out_dir}; nothing to write")
        return
    (out_dir / "encoder_block.json").write_text(
        json.dumps({"tokens": tokens, "constants": dtypes, "graph": graph}, default=str)
    )
    np.savez(out_dir / "encoder_block.npz", **constants)

    n = int(encoder.output_lengths(partial.shape[1]))
    np.savez(
        out_dir / "encoder_reference.npz",
        full_mel=full.astype(np.float32),
        full_bf16=run(encoder, full, mx.bfloat16),
        full_f32=run(encoder, full, mx.float32),
        partial_mel=np.pad(partial, [(0, 0), (0, frames - partial.shape[1])]).astype(np.float32),
        partial_tokens=np.int32(n),
        partial_bf16=run(encoder, partial, mx.bfloat16),
        partial_f32=run(encoder, partial, mx.float32),
    )
    primitives = sum(r["type"] == "primitive" for r in graph)
    weights = sum(c.size for c in constants.values())
    print(f"block: {frames} frames -> {tokens} tokens, {primitives} primitives, "
          f"{weights / 1e6:.0f} M constants -> {out_dir}")
    print(f"partial reference: {partial.shape[1]} frames -> {n} tokens")


if __name__ == "__main__":
    main()
