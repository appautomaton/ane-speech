"""Trace Qwen3-ForcedAligner's text decoder in segments, for conversion.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_aligner_trace [--cache 2048]

The aligner's decoder is Qwen3-ASR's at a smaller width, with a head that maps
each row to one of 5000 time classes instead of a token. It runs once over the
whole prompt, so only the chunk shape is traced, with no one-row step: the
runtime feeds the prompt in chunks against the KV cache, which computes what
one causal pass over it does. The cache bounds the prompt: audio and
transcript take about 25 rows per second of speech, so 2048 rows hold about
80 s. Segments and head are reference/qwen3_asr_decoder_trace.py's functions.

Writes to artifacts/qwen3_asr/qwen3_forced_aligner_0_6b/:
decoder_segment<i>.{json,npz}, decoder_head.{json,npz}, and
decoder_reference.npz (the last reference clip: input embeddings, cos/sin,
the timestamp rows, and mlx-speech's classes and logits there).

References come from mlx-speech in float32. Time classes can nearly tie: a
word ending at a pause may score almost the same at either edge of it, several
classes apart, so bf16 noise alone moves a few of them (mlx-speech's own bf16
pass differs from its float32 one by up to 7 classes). Before writing, the
segmented decoder must reproduce mlx-speech's float32 classes exactly, in
float32, on every reference clip; the script fails otherwise. The traced
weights are bf16, as the checkpoint stores them.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
from mlx_speech.generation.qwen3_forced_aligner import Qwen3ForcedAligner

from convert.qwen3_asr import align_clips, artifacts_dir, model_dir
from reference.qwen3_asr_decoder_trace import (
    CHUNK, MASKED, SEGMENT, head_function, run_segmented, save_trace, segment_function,
)

CHECKPOINT = "qwen3_forced_aligner_0_6b"
CACHE = 2048  # default for --cache


def inputs(aligner: Qwen3ForcedAligner, wav, text: str, language: str):
    """The prompt's input embeddings with the audio in place, RoPE tables, the
    timestamp rows, and mlx-speech's classes and float32 logits there."""
    prepared = aligner.prepare(wav, text, language=language)
    model = aligner.model
    audio = model.get_audio_features(
        mx.array(prepared.features.input_features, dtype=mx.float32),
        feature_attention_mask=mx.array(prepared.features.feature_attention_mask, dtype=mx.int32),
    )
    ids = mx.array([prepared.input_ids], dtype=mx.int32)
    embeds = np.array(model.prepare_inputs_embeds(ids, audio)[0].astype(mx.float32))
    rope = model.text_decoder.model.layers[0].self_attn.rotary_emb
    cos, sin = (np.array(a) for a in rope.cos_sin(seq_len=embeds.shape[0], offset=0))
    logits = np.array(model.text_decoder.project_logits(aligner.timestamp_hidden_states(prepared)).astype(mx.float32))
    positions = np.array(prepared.timestamp_positions, np.int64)
    return embeds, cos, sin, positions, logits.argmax(-1), logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=int, default=CACHE, help=f"KV cache rows, a multiple of {CHUNK}")
    args = parser.parse_args()
    if args.cache <= 0 or args.cache % CHUNK:
        parser.error(f"--cache must be a positive multiple of {CHUNK}")
    cache = args.cache
    out_dir = artifacts_dir(CHECKPOINT)
    out_dir.mkdir(parents=True, exist_ok=True)

    def split(aligner: Qwen3ForcedAligner):
        decoder = aligner.model.text_decoder
        layers, config = decoder.model.layers, aligner.config.text_config
        segments = [segment_function(layers[i : i + SEGMENT], config) for i in range(0, len(layers), SEGMENT)]
        return segments, head_function(decoder)

    reference = Qwen3ForcedAligner.from_dir(model_dir(CHECKPOINT), dtype=mx.float32)
    config = reference.config.text_config
    exact_segments, exact_head = split(reference)
    for wav, text, language in align_clips():
        embeds, cos, sin, positions, classes, logits = inputs(reference, wav, text, language)
        if embeds.shape[0] > cache:
            raise SystemExit(f"{wav.name}: {embeds.shape[0]} rows exceed the cache of {cache}.")
        ours = run_segmented(exact_segments, exact_head, embeds, cos, sin, config, mx.float32, cache)
        same = float((ours[positions].argmax(-1) == classes).mean())
        print(f"{wav.name}: {embeds.shape[0]} rows, {len(positions)} timestamps; "
              f"segmented MLX float32 matches mlx-speech float32 on {same:.1%}")
        if same < 1.0:
            raise SystemExit("The segmented decoder diverges from mlx-speech.")

    segments, head = split(Qwen3ForcedAligner.from_dir(model_dir(CHECKPOINT)))

    bf16 = mx.bfloat16
    kv, d, h = config.num_key_value_heads, config.head_dim, config.hidden_size
    example = (
        mx.zeros((CHUNK, h), bf16), mx.zeros((CHUNK, d), bf16), mx.zeros((CHUNK, d), bf16),
        mx.zeros((SEGMENT, kv, cache, d), bf16), mx.zeros((SEGMENT, kv, cache, d), bf16),
        mx.zeros((CHUNK, cache + CHUNK), bf16),
    )
    meta = {"segment": SEGMENT, "chunk": CHUNK, "cache": cache, "masked": MASKED}
    for i, segment in enumerate(segments):
        count = save_trace(out_dir, f"decoder_segment{i}", segment, *example, meta=meta)
        print(f"decoder_segment{i}: {count} primitives, at T={CHUNK}")
    count = save_trace(out_dir, "decoder_head", head, mx.zeros((CHUNK, h), bf16), meta=meta)
    print(f"decoder_head: {count} primitives")
    np.savez(
        out_dir / "decoder_reference.npz",
        embeds=embeds.astype(np.float32), cos=cos, sin=sin, prompt_rows=np.int32(embeds.shape[0]),
        positions=positions, tokens=classes, logits=logits,
    )


if __name__ == "__main__":
    main()
