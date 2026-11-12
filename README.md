<div align="center">

# ane-speech

**Speech recognition on the Apple Neural Engine**

Local transcription that leaves the GPU free.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Neural%20Engine-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)
[![Swift](https://img.shields.io/badge/runtime-Swift-F05138?logo=swift&logoColor=white)](https://www.swift.org)
[![Core AI](https://img.shields.io/badge/models-Core%20AI-0A84FF)](https://developer.apple.com)
[![GitHub](https://img.shields.io/badge/GitHub-appautomaton-181717?logo=github)](https://github.com/appautomaton)

[Models](#models) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Performance](docs/performance.md)

</div>

ane-speech converts open speech-recognition models into Apple's Core AI format
and runs them on the Neural Engine. Every layer runs there, so transcription
stays quick while the GPU is busy with something else: another model, a
render, a game.

It grew out of [mlx-speech](https://github.com/appautomaton/mlx-speech), whose
models it converts, and powers the on-device recognition in
[TNT](https://github.com/appautomaton/tnt-asr).

## Models

| Model | Best for |
| --- | --- |
| **Confucius4-R2T2** | Live transcription: text appears while you speak |
| **Qwen3-ASR 1.7B** | Whole recordings: one accurate pass once you stop |

Both are multilingual, take a hint of names and terms to expect, and match
their MLX originals word for word apart from rare near-ties.

## Quick start

You need a Mac with Apple silicon and a macOS that includes Core AI; exact
versions live in `Package.swift`. Convert a model once
([Conversion](#conversion)), then:

```bash
git clone https://github.com/appautomaton/ane-speech.git
cd ane-speech
swift build -c release
.build/release/ane-speech transcribe speech.wav --checkpoint qwen3_asr_1_7b
.build/release/ane-speech stream speech.wav
```

`transcribe` reads a whole recording. `stream` plays it through R2T2 as if it
were spoken live, printing text as it settles.

## How it works

```
open checkpoint  ->  trace in MLX  ->  convert to Core AI  ->  run on the Neural Engine
```

Conversion runs once, in Python. It traces each model's MLX graph, rebuilds
it for the Neural Engine in fp16, and checks the result against the original.
From then on a small Swift runtime does the work, reading only its own
converted files, with no Python and no PyTorch.

Other apps can keep a model loaded with `ane-speech serve`, which answers
JSON requests on stdin; TNT uses it this way.

## Conversion

Each model is converted once from its open checkpoint, with
[uv](https://docs.astral.sh/uv/) and a checkout of mlx-speech next to this
one. [AGENTS.md](AGENTS.md) walks through the steps, the checks, and the
options; a coding agent can follow it as written.

## License

The code is [MIT](LICENSE). Converted models keep the licenses of their
sources: Qwen3-ASR under Apache-2.0, Confucius4-R2T2 under the NetEase Youdao
Model Use License Agreement.

## Acknowledgements

The Qwen team for Qwen3-ASR, NetEase Youdao for Confucius4-R2T2, and
[mlx-speech](https://github.com/appautomaton/mlx-speech) for the MLX
implementations this project builds on.
