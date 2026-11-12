"""The Qwen3-ASR audio encoder as one model per 8-second block.

encoder.aimodel, function `encode`:
  in   mel         fp16 [128, 800]   log-mel block, zero past the clip's end
  in   key_mask    fp16 [104]        0 for valid tokens, MASKED for padding
  out  embeddings  fp16 [104, 2048]  rows past the valid tokens are padding

The encoder attends only within 8-second blocks, so a long clip is one call per
block, and a streaming session recomputes only the block still filling.
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

MASKED = -30000.0


def build(out_dir: Path) -> None:
    trace = Trace.load(out_dir / "encoder_block")
    (_, mel_shape), = trace.inputs
    (_, out_shape), = trace.outputs
    tokens = trace.meta["tokens"]

    def encode(mel, key_mask):
        return Translator(trace, key_mask=key_mask)(mel)[0]

    encode.__annotations__ = {
        "mel": Annotated[Value, TensorType(mel_shape, F16, name="mel")],
        "key_mask": Annotated[Value, TensorType((tokens,), F16, name="key_mask")],
        "return": Annotated[Value, TensorType(out_shape, F16, name="embeddings")],
    }
    module = Module.create()
    with module:
        graph(encode)
    AIProgram(module).save_asset(out_dir / "encoder.aimodel")
    meta = {"block_frames": mel_shape[1], "tokens": tokens, "hidden": out_shape[1], "masked": MASKED}
    (out_dir / "encoder.json").write_text(json.dumps(meta))


async def check(out_dir: Path) -> bool:
    ref = np.load(out_dir / "encoder_reference.npz")
    encode, note = await load(out_dir / "encoder.aimodel", "encode")
    print(note)
    tokens = ref["full_f32"].shape[0]
    ok = True
    for case, valid in (("full", tokens), ("partial", int(ref["partial_tokens"]))):
        mask = np.where(np.arange(tokens) < valid, 0.0, MASKED).astype(F16)
        inputs = {"mel": NDArray(ref[f"{case}_mel"].astype(F16)), "key_mask": NDArray(mask)}
        out = (await encode(inputs))["embeddings"].numpy()[:valid]
        core, bf16 = rel_l2(out, ref[f"{case}_f32"]), rel_l2(ref[f"{case}_bf16"], ref[f"{case}_f32"])
        good = bool(np.isfinite(out).all()) and core <= bf16
        ok &= good
        print(f"  {case:7s} {valid:3d} tokens  rel L2 vs MLX f32: Core AI fp16 {core:.2e}, "
              f"MLX bf16 {bf16:.2e}  {'ok' if good else 'FAIL'}")

    inputs = {"mel": NDArray(ref["full_mel"].astype(F16)), "key_mask": NDArray(np.zeros(tokens, F16))}
    calls = 0

    def work(seconds: float) -> None:
        nonlocal calls
        loop = asyncio.new_event_loop()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            loop.run_until_complete(encode(inputs))
            calls += 1
        loop.close()

    idle, busy = await asyncio.to_thread(ane_activity, 3.0, work)
    print(f"  ANE active {idle:.0f}% idle, {busy:.0f}% while encoding; {3e3 / calls:.1f} ms per 8 s block")
    return ok and busy > 50
