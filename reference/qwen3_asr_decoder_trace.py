"""Trace the Qwen3-ASR text decoder in segments, for conversion.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_trace [--cache 1024]

The decoder is split into segments of SEGMENT layers plus a head, each traced
over a fixed chunk of CHUNK tokens against a KV cache of `--cache` positions. A
segment is a pure function; the caller owns the cache and writes the returned
rows into it:

  segment(x [T, H], cos [T, D], sin [T, D],
          k_cache [n, KV, L, D], v_cache [n, KV, L, D], mask [T, L + T])
      -> x [T, H], k_new [n, KV, T, D], v_new [n, KV, T, D]
  head(x [T, H]) -> logits [T, vocab]

Each segment is traced twice, at T = CHUNK for prompts and at T = 1 for
generating a token, and both traces share one file of weights. A pass costs a
fixed part, reading the weights, plus a part that grows with T: on the ANE a
segment takes 10.1 ms at T=128 and 5.7 ms at T=1 (docs/performance.md). R2T2
re-prefills the open audio block and all text each window, a few hundred rows,
so the wide shape suits prompts; a token generated alone pays only the floor.

The cache bounds a streaming take: R2T2 keeps the whole take in the prompt,
about 16 rows per second of speech, so 1024 rows hold about a minute. Every
pass attends over the whole cache, full or not: 2048 rows cost about 45% more
per pass than 1024 (docs/performance.md). The runtime reads the size from
decoder.json, so changing it needs only this flag and a reconversion.

`mask` is additive: 0 where a row may attend, a large negative value elsewhere.
Positions enter only through cos/sin, so one graph serves every position.
Segments use mlx-speech's own projections and MLP; RMSNorm is the fused
`mx.fast.rms_norm`, which the translator lowers in an fp16-safe form.

Writes to artifacts/qwen3_asr/<checkpoint>/: decoder_segment<i>.{json,npz},
decoder_segment<i>_step.json (its weights are decoder_segment<i>.npz),
decoder_head.{json,npz}, and decoder_reference.npz (test.wav, teacher-forced: input embeddings, cos/sin,
the reference tokens, and mlx-speech's float32 logits at every decode step).
Before writing, the segmented decoder must reproduce mlx-speech's greedy tokens
in MLX; the script fails otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx_speech.models.qwen3_asr.text_decoder as td
import numpy as np
from mlx_speech.generation.qwen3_asr import Qwen3ASRTranscriber
from mlx_speech.models.qwen3_asr.model import replace_audio_embeddings

from convert.qwen3_asr import add_checkpoint_argument, artifacts_dir, model_dir, sample_wav

SEGMENT, CHUNK, CACHE = 7, 128, 1152  # CACHE: default for --cache
STEP = 1  # rows per pass when generating a token
MASKED = -30000.0


def rms(x: mx.array, norm: td.Qwen3ASRTextRMSNorm) -> mx.array:
    return mx.fast.rms_norm(x, norm.weight.astype(x.dtype), norm.eps)


def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    return x * cos + td._rotate_half(x) * sin


def segment_function(layers: list, config) -> callable:
    heads, kv, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    group = heads // kv

    def segment(x, cos, sin, k_cache, v_cache, mask):
        t = x.shape[0]
        new_k, new_v = [], []
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            h = rms(x, layer.input_layernorm)
            q = rms(attn.q_proj(h).reshape(t, heads, d), attn.q_norm).transpose(1, 0, 2)
            k = rms(attn.k_proj(h).reshape(t, kv, d), attn.k_norm).transpose(1, 0, 2)
            v = attn.v_proj(h).reshape(t, kv, d).transpose(1, 0, 2)
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
            new_k.append(k)
            new_v.append(v)
            keys = mx.concatenate([k_cache[i], k], axis=1)
            values = mx.concatenate([v_cache[i], v], axis=1)
            # Query heads g*j .. g*j+g-1 share KV head j: fold them into the rows.
            qg = q.reshape(kv, group * t, d)
            scores = (qg @ keys.transpose(0, 2, 1)) * attn.scale + mx.tile(mask, (group, 1))
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
            out = (probs @ values).reshape(heads, t, d).transpose(1, 0, 2).reshape(t, heads * d)
            x = x + attn.o_proj(out)
            x = x + layer.mlp(rms(x, layer.post_attention_layernorm), native=True)
        return x, mx.stack(new_k), mx.stack(new_v)

    return segment


def head_function(decoder) -> callable:
    def head(x):
        return decoder.lm_head(rms(x, decoder.model.norm))

    return head


def chunk_inputs(embeds: np.ndarray, cos: np.ndarray, sin: np.ndarray, start: int, filled: int, cache: int):
    """Inputs for the chunk starting at `start`, with `filled` cache rows valid."""
    n = min(CHUNK, embeds.shape[0] - start)
    pad = ((0, CHUNK - n), (0, 0))
    mask = np.full((CHUNK, cache + CHUNK), MASKED, np.float32)
    mask[:, :filled] = 0.0
    mask[:, cache:] = np.where(np.tril(np.ones((CHUNK, CHUNK), bool)), 0.0, MASKED)
    rows = slice(start, start + n)
    return np.pad(embeds[rows], pad), np.pad(cos[rows], pad), np.pad(sin[rows], pad), mask, n


def run_segmented(segments, head, embeds, cos, sin, config, dtype, cache: int) -> np.ndarray:
    """Teacher-forced pass over all rows; returns logits for every row."""
    layers, kv, d = config.num_hidden_layers, config.num_key_value_heads, config.head_dim
    k_cache = np.zeros((layers, kv, cache, d), np.float32)
    v_cache = np.zeros_like(k_cache)
    logits = []
    for start in range(0, embeds.shape[0], CHUNK):
        x, c, s, mask, n = chunk_inputs(embeds, cos, sin, start, start, cache)
        x = mx.array(x, dtype=dtype)
        args = [mx.array(a, dtype=dtype) for a in (c, s)]
        for i, segment in enumerate(segments):
            span = slice(i * SEGMENT, (i + 1) * SEGMENT)
            x, k_new, v_new = segment(
                x, *args, mx.array(k_cache[span], dtype=dtype), mx.array(v_cache[span], dtype=dtype),
                mx.array(mask, dtype=dtype),
            )
            k_cache[span, :, start : start + n] = np.array(k_new.astype(mx.float32))[:, :, :n]
            v_cache[span, :, start : start + n] = np.array(v_new.astype(mx.float32))[:, :, :n]
        logits.append(np.array(head(x).astype(mx.float32))[:n])
    return np.concatenate(logits)


def save_trace(out_dir: Path, name: str, fn, *example: mx.array, meta: dict, weights: str | None = None) -> int:
    """Writes `<name>.json`, the graph, and `<name>.npz`, its constants. With
    `weights`, the constants must equal that earlier trace's exactly, and the
    graph points at its file instead of writing another."""
    records: list[dict] = []
    mx.export_function(records.append, fn, *example)
    constants, dtypes, graph = {}, {}, []
    for record in records:
        if record["type"] != "constants":
            graph.append(record)
            continue
        for key, value in record["constants"]:
            dtypes[key] = str(value.dtype).removeprefix("mlx.core.")
            constants[key] = np.array(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)
    doc = {**meta, "constants": dtypes, "graph": graph}
    if weights is None:
        np.savez(out_dir / f"{name}.npz", **constants)
    else:
        stored = np.load(out_dir / f"{weights}.npz")
        if set(stored.files) != set(constants) or not all(np.array_equal(stored[k], v) for k, v in constants.items()):
            raise SystemExit(f"{name}: its constants differ from {weights}'s")
        doc["weights"] = weights
    (out_dir / f"{name}.json").write_text(json.dumps(doc, default=str))
    return sum(r["type"] == "primitive" for r in graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_checkpoint_argument(parser)
    parser.add_argument("--wav", type=Path, default=sample_wav())
    parser.add_argument("--cache", type=int, default=CACHE, help=f"KV cache rows, a multiple of {CHUNK}")
    args = parser.parse_args()
    if args.cache <= 0 or args.cache % CHUNK:
        parser.error(f"--cache must be a positive multiple of {CHUNK}")
    cache = args.cache
    out_dir = artifacts_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = Qwen3ASRTranscriber.from_dir(model_dir(args.checkpoint))
    model, decoder, config = asr.model, asr.model.text_decoder, asr.config.text_config
    mx.eval(model.parameters())

    # mlx-speech's reference transcription, keeping its float32 logits per step.
    steps: list[np.ndarray] = []
    for method in ("prefill", "decode_step"):
        original = getattr(td.Qwen3ASRTextForCausalLM, method)

        def keep(self, *a, _original=original, **k):
            out = _original(self, *a, **k)
            steps.append(np.array(out.logits[0, -1].astype(mx.float32)))
            return out

        setattr(td.Qwen3ASRTextForCausalLM, method, keep)
    result = asr.transcribe(args.wav)
    tokens = np.array(result.tokens, np.int64)

    # The same inputs, rebuilt: prompt with audio features, then the generated tokens.
    batch = asr.processor.feature_extractor(args.wav)
    prompt = asr.processor.build_prompt(context="", audio_length=int(batch.audio_lengths.reshape(-1)[0]))
    audio = model.get_audio_features(
        mx.array(batch.input_features, dtype=mx.float32),
        feature_attention_mask=mx.array(batch.feature_attention_mask, dtype=mx.int32),
    )
    ids = mx.array([prompt.input_ids], dtype=mx.int32)
    embeds = replace_audio_embeddings(ids, model.embed_input_ids(ids), audio, audio_token_id=asr.config.audio_token_id)
    embeds = mx.concatenate([embeds[0], decoder.model.embed_tokens(mx.array(tokens[:-1]))]).astype(mx.float32)
    p = len(prompt.input_ids)
    n = embeds.shape[0]
    if p + len(tokens) - 1 + CHUNK > cache + CHUNK:
        raise SystemExit(f"{n} rows exceed the cache of {cache}.")
    rope = decoder.model.layers[0].self_attn.rotary_emb
    cos, sin = (np.array(a) for a in rope.cos_sin(seq_len=n, offset=0))
    embeds = np.array(embeds)

    layers = decoder.model.layers
    segments = [segment_function(layers[i : i + SEGMENT], config) for i in range(0, len(layers), SEGMENT)]
    head = head_function(decoder)
    logits = run_segmented(segments, head, embeds, cos, sin, config, mx.bfloat16, cache)
    predicted = logits[p - 1 :].argmax(-1)
    agree = float((predicted == tokens).mean())
    print(f"{args.wav.name}: {p} prompt rows, {len(tokens)} tokens; "
          f"segmented MLX bf16 agrees with mlx-speech on {agree:.0%} of greedy tokens")
    if agree < 1.0:
        raise SystemExit("The segmented decoder diverges from mlx-speech.")

    bf16 = mx.bfloat16
    kv, d, h = config.num_key_value_heads, config.head_dim, config.hidden_size

    def example(t: int) -> tuple[mx.array, ...]:
        return (
            mx.zeros((t, h), bf16), mx.zeros((t, d), bf16), mx.zeros((t, d), bf16),
            mx.zeros((SEGMENT, kv, cache, d), bf16), mx.zeros((SEGMENT, kv, cache, d), bf16),
            mx.zeros((t, cache + t), bf16),
        )

    meta = {"segment": SEGMENT, "chunk": CHUNK, "cache": cache, "masked": MASKED}
    for i, segment in enumerate(segments):
        name = f"decoder_segment{i}"
        count = save_trace(out_dir, name, segment, *example(CHUNK), meta=meta)
        save_trace(out_dir, f"{name}_step", segment, *example(STEP), meta={**meta, "chunk": STEP}, weights=name)
        print(f"{name}: {count} primitives, at T={CHUNK} and T={STEP}")
    count = save_trace(out_dir, "decoder_head", head, mx.zeros((CHUNK, h), bf16), meta=meta)
    print(f"decoder_head: {count} primitives")
    np.savez(
        out_dir / "decoder_reference.npz",
        embeds=embeds.astype(np.float32), cos=cos, sin=sin, prompt_rows=np.int32(p),
        tokens=tokens, logits=np.stack(steps)[: len(tokens)].astype(np.float32),
    )
    print(f"reference: {result.text}")


if __name__ == "__main__":
    main()
