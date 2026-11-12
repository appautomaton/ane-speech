---
language:
- en
- zh
license: apache-2.0
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ForcedAligner-0.6B
tags:
- core-ai
- apple-neural-engine
- apple-silicon
- forced-alignment
- timestamps
- speech
- qwen3
---

# Qwen3-ForcedAligner-0.6B — Apple Neural Engine (fp16)

[![GitHub](https://img.shields.io/badge/GitHub-ane--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/ane-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

Core AI conversion of [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) that runs entirely on the Apple Neural Engine. Given speech and its exact transcript, it returns when every Chinese character and every English word was spoken, while the GPU stays free for something else. It runs through the [`ane-speech`](https://github.com/appautomaton/ane-speech) Swift runtime, with no Python, PyTorch, or cloud API at inference time.

## Model Details

- Developed by: [App Automaton](https://appautomaton.com)
- Upstream model: [`Qwen/Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (code: [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR))
- Architecture: Qwen3-ASR's audio encoder and a 0.6B text decoder whose head sorts each timestamp into one of 5000 classes of 80 ms; one pass times the whole transcript
- Conversion: from the official bf16 weights, by tracing the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) MLX graph and rebuilding it with Core AI; no PyTorch involved
- Precision: fp16 weights and activations. Activation ranges were measured before conversion and stay well inside fp16.
- Placement: every layer runs on the Neural Engine
- Parity: time classes match mlx-speech's float32 pass on our test clips; a bf16 pass moves a few near-ties
- Input: 16 kHz mono audio and its exact transcript, Chinese, English, or mixed
- Requirements: a Mac with Apple silicon and macOS 27 or later

## Files

| Path | Contents |
| --- | --- |
| `qwen3_forced_aligner_0_6b/` | the text decoder in four segments and the time-class head; `audio_encoder/` holds its encoder, and `runtime/` the embedding table, tokenizer, and audio front-end constants |

## How to Get Started

The aligner runs from a source checkout of [`ane-speech`](https://github.com/appautomaton/ane-speech); its PyPI release carries the recognizers. The model downloads from this repo on first use.

```bash
git clone https://github.com/appautomaton/ane-speech.git
cd ane-speech
swift build -c release
uv run ane-speech align speech.wav --text "the exact transcript of speech.wav"   # start, end, and unit per line
```

From Python, in the checkout (`uv run python`):

```python
import ane_speech

aligner = ane_speech.load("qwen3_forced_aligner_0_6b")
for text, start_s, end_s in aligner.generate("speech.wav", "the exact transcript", language="en"):
    print(f"{start_s:.2f} {end_s:.2f} {text}")
```

For a timed transcript, transcribe with [Qwen3-ASR](https://huggingface.co/appautomaton/qwen3-asr-1.7b-fp16-ane) first, then align its text. The first load compiles the model for your Mac's Neural Engine and caches the result, so later loads take a fraction of a second.

## Reading the Times

- Chinese characters are units one by one. English words keep only letters, numbers, and apostrophes, so `3.75%` becomes `375`. Punctuation makes no unit: a pause at a comma shows as a gap between its neighbours.
- Times come in 80 ms classes. Where predictions run backwards, upstream's repair keeps the longest non-decreasing run, snaps isolated outliers to a neighbour, and interpolates longer runs, so a repaired time can fall between classes.
- The transcript must match the speech: every unit gets a span, whether it was spoken or not.

## Notes

- Each call takes about 75 s of Chinese or 90 s of English with its transcript; split longer audio. (Upstream accepts 180 s; this conversion's cache holds less.)
- Chinese and English only, as in mlx-speech.
- This repo contains Core AI models only, with no PyTorch or safetensors checkpoint.

## Links

- Source code: [`appautomaton/ane-speech`](https://github.com/appautomaton/ane-speech)
- Recognizers: [`appautomaton/qwen3-asr-1.7b-fp16-ane`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-fp16-ane) · [`appautomaton/confucius4-r2t2-fp16-ane`](https://huggingface.co/appautomaton/confucius4-r2t2-fp16-ane)
- More from App Automaton: [Project](https://appautomaton.com) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

Apache 2.0, following the upstream [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) release; a copy is included as [`LICENSE`](LICENSE). The `ane-speech` runtime is licensed separately under MIT.
