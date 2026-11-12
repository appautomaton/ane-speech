"""The Qwen3-ASR text decoder as layer segments plus a head, each its own model.

decoder_segment<i>.aimodel, function `segment` (n layers, chunk T, cache L):
  in   hidden      fp16 [T, 2048]          rows past the valid ones are padding
  in   cos, sin    fp16 [T, 128]           RoPE for each row's position
  in   k_cache     fp16 [n, 8, L, 128]     this segment's cached keys
  in   v_cache     fp16 [n, 8, L, 128]
  in   mask        fp16 [T, L + T]         additive: 0 may attend, MASKED may not
  out  hidden_out  fp16 [T, 2048]
  out  k_new       fp16 [n, 8, T, 128]     the caller writes these into the cache
  out  v_new       fp16 [n, 8, T, 128]

decoder_head.aimodel, function `head`: hidden [T, 2048] -> logits [T, vocab].

Prefill runs the prompt through in chunks of T; each generated token is one more
chunk with one valid row. The weights dominate a call, so a chunk of T=128
costs about 1.5x a chunk of 32 (see reference/qwen3_asr_decoder_trace.py).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from coreai.authoring import AIProgram, Module, TensorType, Value, graph
from coreai.runtime import NDArray

from convert.qwen3_asr.check import ane_activity, load, rel_l2
from convert.qwen3_asr.mlx_graph import F16, Trace, Translator
SEGMENT_IN = ("hidden", "cos", "sin", "k_cache", "v_cache", "mask")
SEGMENT_OUT = ("hidden_out", "k_new", "v_new")


def tensor(shape, name: str):
    return Annotated[Value, TensorType(tuple(shape), F16, name=name)]


def segment_stems(out_dir: Path) -> list[Path]:
    """decoder_segment<i>, in order; each may have a decoder_segment<i>_step trace."""
    stems = [p.with_suffix("") for p in out_dir.glob("decoder_segment*.json") if p.stem.removeprefix("decoder_segment").isdigit()]
    return sorted(stems, key=lambda p: int(p.name.removeprefix("decoder_segment")))


def step_stem(stem: Path) -> Path | None:
    step = stem.with_name(stem.name + "_step")
    return step if step.with_suffix(".json").exists() else None


def _segment_function(trace: Trace):
    def segment(hidden, cos, sin, k_cache, v_cache, mask):
        return tuple(Translator(trace)(hidden, cos, sin, k_cache, v_cache, mask))

    segment.__annotations__ = {n: tensor(s, n) for n, (_, s) in zip(SEGMENT_IN, trace.inputs, strict=True)}
    segment.__annotations__["return"] = tuple[
        tuple(tensor(s, n) for n, (_, s) in zip(SEGMENT_OUT, trace.outputs, strict=True))
    ]
    return segment


def build(out_dir: Path) -> None:
    """One model per segment: function `segment` at the trace's T and, when
    traced, `step` at one row, so the two shapes share the weights on disk."""
    for stem in segment_stems(out_dir):
        module = Module.create()
        with module:
            graph(_segment_function(Trace.load(stem)))
            if step := step_stem(stem):
                graph(_segment_function(Trace.load(step)), provided_name="step")
        AIProgram(module).save_asset(stem.with_suffix(".aimodel"))

    trace = Trace.load(out_dir / "decoder_head")
    (_, in_shape), = trace.inputs
    (_, out_shape), = trace.outputs

    def head(hidden):
        return Translator(trace)(hidden)[0]

    head.__annotations__ = {"hidden": tensor(in_shape, "hidden"), "return": tensor(out_shape, "logits")}
    module = Module.create()
    with module:
        graph(head)
    AIProgram(module).save_asset(out_dir / "decoder_head.aimodel")
    export_meta(out_dir)
    export_reference(out_dir)


def export_meta(out_dir: Path) -> None:
    """decoder.json: the shapes the Swift runtime needs to drive the segments."""
    meta = json.loads((out_dir / "decoder_segment0.json").read_text())
    stems = segment_stems(out_dir)
    step = all(step_stem(stem) for stem in stems)
    (out_dir / "decoder.json").write_text(json.dumps({
        "segments": len(stems),
        **{k: meta[k] for k in ("segment", "chunk", "cache", "masked")},
        **({"step": 1} if step else {}),
    }))


def export_reference(out_dir: Path) -> None:
    """decoder_reference.npz as raw little-endian files the Swift check reads
    directly. The aligner's reference is checked here only."""
    ref = np.load(out_dir / "decoder_reference.npz")
    if "positions" in ref:
        return
    meta = json.loads((out_dir / "decoder_segment0.json").read_text())
    ref_dir = out_dir / "decoder_reference"
    ref_dir.mkdir(exist_ok=True)
    for name in ("embeds", "cos", "sin"):
        ref[name].astype("<f2").tofile(ref_dir / f"{name}.f16")
    (ref_dir / "meta.json").write_text(json.dumps({
        "rows": int(ref["embeds"].shape[0]),
        "prompt_rows": int(ref["prompt_rows"]),
        "tokens": ref["tokens"].tolist(),
        **{k: meta[k] for k in ("segment", "chunk", "cache", "masked")},
    }))


class Decoder:
    """Runs the segment and head models over rows of input embeddings, owning the
    KV cache. A single row goes through the one-row `step` functions when given,
    as the Swift runtime does."""

    def __init__(self, segments, head, meta: dict, layers: int, kv: int, d: int, steps=None) -> None:
        self.segments, self.head, self.meta, self.steps = segments, head, meta, steps
        self.k = np.zeros((layers, kv, meta["cache"], d), F16)
        self.v = np.zeros_like(self.k)
        self.filled = 0

    def _inputs(self, embeds, cos, sin, t: int):
        cache, masked = self.meta["cache"], self.meta["masked"]
        n = embeds.shape[0]
        pad = ((0, t - n), (0, 0))
        mask = np.full((t, cache + t), masked, F16)
        mask[:, : self.filled] = 0
        mask[:, cache:] = np.where(np.tril(np.ones((t, t), bool)), 0, masked)
        return [np.pad(a, pad).astype(F16) for a in (embeds, cos, sin)] + [mask]

    async def chunk(self, embeds: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Feed up to `chunk` rows; returns their logits and appends their keys and values."""
        n, size, wide = embeds.shape[0], self.meta["segment"], self.meta["chunk"]
        one = n == 1 and self.steps is not None
        hidden, c, s, mask = self._inputs(embeds, cos, sin, 1 if one else wide)
        rows = slice(self.filled, self.filled + n)
        for i, segment in enumerate(self.steps if one else self.segments):
            span = slice(i * size, (i + 1) * size)
            out = await segment({
                "hidden": NDArray(hidden), "cos": NDArray(c), "sin": NDArray(s),
                "k_cache": NDArray(self.k[span]), "v_cache": NDArray(self.v[span]), "mask": NDArray(mask),
            })
            hidden = out["hidden_out"].numpy().copy()
            self.k[span, :, rows] = out["k_new"].numpy()[:, :, :n]
            self.v[span, :, rows] = out["v_new"].numpy()[:, :, :n]
        self.filled += n
        hidden = np.pad(hidden, ((0, wide - hidden.shape[0]), (0, 0)))  # the head takes `chunk` rows
        return (await self.head({"hidden": NDArray(hidden)}))["logits"].numpy()[:n].astype(np.float32)


async def check(out_dir: Path) -> bool:
    ref = np.load(out_dir / "decoder_reference.npz")
    meta = json.loads((out_dir / "decoder_segment0.json").read_text())
    meta = {k: meta[k] for k in ("segment", "chunk", "cache", "masked")}
    segments, step_functions = [], []
    for stem in segment_stems(out_dir):
        segment, note = await load(stem.with_suffix(".aimodel"), "segment")
        segments.append(segment)
        print(note)
        if step_stem(stem):
            step_functions.append((await load(stem.with_suffix(".aimodel"), "step"))[0])
    head, note = await load(out_dir / "decoder_head.aimodel", "head")
    print(note)

    embeds, cos, sin = ref["embeds"], ref["cos"], ref["sin"]
    layers = len(segments) * meta["segment"]
    kv, d = 8, cos.shape[1]
    decoder = Decoder(segments, head, meta, layers, kv, d, step_functions or None)
    # As in generation: the prompt in chunks, then each generated token alone.
    # The aligner has no generated tokens: its classes are read at `positions`.
    p, tokens, t = int(ref["prompt_rows"]), ref["tokens"], meta["chunk"]
    spans = [(i, min(i + t, p)) for i in range(0, p, t)] + [(i, i + 1) for i in range(p, embeds.shape[0])]
    start = time.perf_counter()
    logits = np.concatenate([await decoder.chunk(embeds[a:b], cos[a:b], sin[a:b]) for a, b in spans])
    took = time.perf_counter() - start
    aligner = "positions" in ref
    steps = logits[ref["positions"]] if aligner else logits[p - 1 : p - 1 + len(tokens)]
    off = np.abs(steps.argmax(-1) - tokens)
    agree = float((off == 0).mean())
    err = rel_l2(steps, ref["logits"])
    finite = bool(np.isfinite(logits).all())
    what = f"time classes agree {agree:.1%}, off by at most {off.max()}" if aligner else f"greedy tokens agree {agree:.0%}"
    print(f"  teacher-forced {embeds.shape[0]} rows ({'tokens one row at a time' if step_functions else 'all in chunks'}) "
          f"in {took:.2f} s: {what} ({len(tokens)}), logits rel L2 vs MLX f32 {err:.2e}, finite {finite}")
    # Near-ties between neighbouring time classes may flip; tokens may not.
    matches = off.max() <= 1 and agree >= 0.98 if aligner else agree == 1.0

    one = [embeds[-1:], cos[-1:], sin[-1:]]
    calls = 0

    def work(seconds: float) -> None:
        nonlocal calls
        loop = asyncio.new_event_loop()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            decoder.filled = p  # decode the same position again and again
            loop.run_until_complete(decoder.chunk(*one))
            calls += 1
        loop.close()

    idle, busy = await asyncio.to_thread(ane_activity, 3.0, work)
    print(f"  ANE active {idle:.0f}% idle, {busy:.0f}% while decoding; {3e3 / calls:.0f} ms per token")
    return matches and finite and busy > 50
