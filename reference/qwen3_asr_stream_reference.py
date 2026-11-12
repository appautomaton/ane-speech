"""Record mlx-speech's R2T2 streaming run, window by window, for the Swift port.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_stream_reference [wav ...]

Feeds each clip in 160 ms pieces through a stream session and writes
artifacts/qwen3_asr/runtime/stream_<compute>.json: per clip,
every step's prefix, budget, prompt length, generated ids, completion,
hypothesis, and committed delta, then the final transcript.

`--compute default` is what mlx-speech ships: bf16 audio and residual stream,
incremental reuse. `--compute reference` is its precision oracle: float32
activations over the bf16 weights, full refeed every window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mlx_speech.asr._adapters.confucius4_r2t2 import (
    DEFAULT_STREAM_COMPUTE,
    REFERENCE_STREAM_COMPUTE,
    Confucius4R2T2Adapter,
)
from mlx_speech.audio.io import load_audio
from mlx_speech.models.confucius4_r2t2.metrics import R2T2StreamTrace

from convert.qwen3_asr import artifacts_dir, model_dir
from reference.qwen3_asr_runtime_assets import CLIPS

OUT = artifacts_dir("confucius4_r2t2") / "runtime"  # streaming is R2T2's alone
COMPUTE = {"default": DEFAULT_STREAM_COMPUTE, "reference": REFERENCE_STREAM_COMPUTE}
PIECE = 2560  # 160 ms at 16 kHz


def run(adapter: Confucius4R2T2Adapter, wav: Path, compute, options: dict) -> dict:
    samples, _ = load_audio(wav, sample_rate=16000)
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    trace = R2T2StreamTrace()
    session = adapter.stream_session(trace=trace, compute=compute, **options)
    state = session._state
    calls: list[dict] = []
    generate = state._llm_generate

    def logged(prefix: str, samples: np.ndarray, budget: int) -> str:
        completion = generate(prefix, samples, budget)
        calls.append({"prefix": prefix, "samples": int(samples.shape[0]), "budget": budget})
        return completion

    state._llm_generate = logged
    for start in range(0, audio.shape[0], PIECE):
        session.feed(audio[start : start + PIECE])
    final = session.finalize()
    assert len(calls) == len(trace.steps)
    steps = []
    for call, step in zip(calls, trace.steps):
        steps.append(
            {
                **call,
                "kind": step.kind,
                "prompt_tokens": step.prompt_tokens,
                "audio_tokens": step.audio_tokens,
                "generated_ids": list(step.generated_ids),
                "completion": step.completion,
                "hypothesis": step.hypothesis,
                "delta": step.delta,
                "committed": step.committed,
            }
        )
    return {"wav": str(wav.resolve()), "session": options, "samples": int(audio.shape[0]), "steps": steps, "text": final.text, "language": final.language}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", type=Path, nargs="*", default=CLIPS)
    parser.add_argument("--model-dir", type=Path, default=model_dir("confucius4_r2t2"))
    parser.add_argument("--compute", choices=COMPUTE, default="default")
    parser.add_argument("--language")
    parser.add_argument("--context", default="")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, default=160)
    parser.add_argument("--unfixed", type=int, default=1, help="unfixed_token_num")
    parser.add_argument("--name", help="output stream_<name>.json (default: the compute name)")
    args = parser.parse_args()
    options = {
        "language": args.language,
        "context": args.context,
        "chunk_ms": args.chunk_ms,
        "lookahead_ms": args.lookahead_ms,
        "unfixed_token_num": args.unfixed,
    }
    adapter = Confucius4R2T2Adapter.from_dir(args.model_dir)
    clips = {}
    for wav in args.wavs:
        clips[wav.stem] = run(adapter, wav, COMPUTE[args.compute], options)
        clip = clips[wav.stem]
        print(f"{wav.stem}: {len(clip['steps'])} steps -> [{clip['language']}] {clip['text']}")
    out = OUT / f"stream_{args.name or args.compute}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clips, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
