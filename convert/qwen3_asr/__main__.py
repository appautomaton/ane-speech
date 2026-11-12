"""Build the Qwen3-ASR models from reference/ traces, then check them on the ANE.

    uv run --project ../mlx-speech python -m reference.qwen3_asr_encoder_trace [--checkpoint <name>]
    uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_trace [--checkpoint <name>]
    uv run python -m convert.qwen3_asr [encoder] [decoder] [--checkpoint <name>] [--no-check]

Loading needs Core AI, so run the last command outside any sandbox. Each check
loads the models preferring the ANE, reports whether MPSGraph placed them fully
on it, compares them with MLX, and measures ANE activity while they run.
"""

from __future__ import annotations

import convert  # noqa: F401  (confines TMPDIR before Core AI loads)

import argparse
import asyncio
import time

from convert.qwen3_asr import add_checkpoint_argument, artifacts_dir, decoder, encoder, encoder_dir

PARTS = {"encoder": encoder, "decoder": decoder}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parts", nargs="*", metavar="{encoder,decoder}", help="default: both")
    add_checkpoint_argument(parser)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    if unknown := set(args.parts) - set(PARTS):
        parser.error(f"unknown part: {', '.join(sorted(unknown))}")
    failed = []
    for name in args.parts or PARTS:
        part = PARTS[name]
        out = encoder_dir() if part is encoder else artifacts_dir(args.checkpoint)  # the encoder is shared
        start = time.perf_counter()
        part.build(out)
        print(f"{name}: built in {time.perf_counter() - start:.1f} s")
        if not args.no_check and not asyncio.run(part.check(out)):
            failed.append(name)
    if failed:
        raise SystemExit(f"check failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
