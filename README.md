<div align="center">

# ane-speech

**Speech recognition on the Apple Neural Engine**

Local transcription that leaves the GPU free.

[![PyPI](https://img.shields.io/pypi/v/ane-speech?color=1f6feb)](https://pypi.org/project/ane-speech/)
[![CI](https://github.com/appautomaton/ane-speech/actions/workflows/ci.yml/badge.svg)](https://github.com/appautomaton/ane-speech/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Neural%20Engine-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)
[![Swift](https://img.shields.io/badge/runtime-Swift-F05138?logo=swift&logoColor=white)](https://www.swift.org)
[![Core AI](https://img.shields.io/badge/models-Core%20AI-0A84FF)](https://developer.apple.com)

[Install](#install) · [Models](#models) · [Command line](#command-line) · [Python](#python) · [How it works](#how-it-works)

</div>

ane-speech runs open speech-recognition models on the Neural Engine of Apple
silicon Macs. Every layer runs there, so transcription stays quick while the
GPU is busy with something else: another model, a render, a game. Nothing
leaves your Mac.

It grew out of [mlx-speech](https://github.com/appautomaton/mlx-speech), whose
models it converts, and powers the on-device recognition in
[TNT](https://github.com/appautomaton/tnt-asr).

## Install

```bash
uv tool install ane-speech     # the command line
pip install ane-speech         # the Python library, command line included
```

You need a Mac with Apple silicon and a macOS release that includes Core AI;
on an older macOS, ane-speech says so when it starts.

## Models

| Name | Model | Best for |
| --- | --- | --- |
| `confucius4_r2t2` (default) | [Confucius4-R2T2](https://huggingface.co/appautomaton/confucius4-r2t2-fp16-ane) | Live transcription: text appears while you speak |
| `qwen3_asr_1_7b` | [Qwen3-ASR 1.7B](https://huggingface.co/appautomaton/qwen3-asr-1.7b-fp16-ane) | Whole recordings: one accurate pass once you stop |

Both are multilingual, take a hint of names and terms to expect, and match
their MLX originals word for word apart from rare near-ties. Each downloads
from Hugging Face the first time you use it. A recording or live take holds
up to about a minute of speech.

## Command line

```bash
ane-speech transcribe meeting.wav --checkpoint qwen3_asr_1_7b
ane-speech transcribe meeting.wav --checkpoint qwen3_asr_1_7b --language Chinese --context "Acme, Kubernetes"
ane-speech stream meeting.wav       # R2T2, played as if spoken live, text printed as it settles
ane-speech pull qwen3_asr_1_7b      # download ahead of time
```

## Python

```python
import ane_speech

asr = ane_speech.load("qwen3_asr_1_7b")
result = asr.generate("meeting.wav")         # a file, or 16 kHz mono float32 samples
print(result.language, result.text)
```

Live, from a microphone:

```python
live = ane_speech.load("confucius4_r2t2")
session = live.stream_session(chunk_ms=480)
for pcm in microphone:                        # 16 kHz mono float32, any length
    print(session.feed(pcm).committed)        # final text, which only grows
print(session.finalize().text)
```

A loaded model runs in a helper process of its own and exits with yours. The
method names follow mlx-speech, so code moves between the two.

## Disk space

Each model takes space twice, about its own size each time: the download, in
the Hugging Face cache, and a copy compiled for your Mac's Neural Engine. The
first load compiles it, which takes a while once, and a macOS update compiles
it again. To remove both:

```bash
rm -rf ~/.cache/huggingface/hub/models--appautomaton--*-ane
rm -rf ~/Library/Caches/coreai-cache/*/ane-speech
```

## How it works

```
open checkpoint  ->  trace in MLX  ->  convert to Core AI  ->  run on the Neural Engine
```

Each model is converted once, in Python: its MLX graph is traced, rebuilt for
the Neural Engine in fp16, and checked against the original. From then on a
small Swift runtime does the work, reading only the converted files, with no
PyTorch. The Python package is a thin layer over that runtime, and other apps
can use the runtime the same way through `ane-speech serve`, which answers
JSON requests on stdin; TNT does.

## Building from source

```bash
git clone https://github.com/appautomaton/ane-speech.git
cd ane-speech
swift build -c release
uv run ane-speech transcribe speech.wav    # models still come from Hugging Face
```

[AGENTS.md](AGENTS.md) covers converting models yourself, the checks behind
them, and releases; a coding agent can follow it as written.

## License

The code is [MIT](LICENSE). The models keep the licenses of their sources:
Qwen3-ASR under Apache-2.0, Confucius4-R2T2 under the NetEase Youdao Model Use
License Agreement.

## Acknowledgements

The Qwen team for Qwen3-ASR, NetEase Youdao for Confucius4-R2T2, and
[mlx-speech](https://github.com/appautomaton/mlx-speech) for the MLX
implementations this project builds on.
