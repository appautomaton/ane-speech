---
language:
- en
- zh
license: apache-2.0
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ASR-1.7B
tags:
- core-ai
- apple-neural-engine
- apple-silicon
- asr
- speech-recognition
- transcription
- qwen3
---

# Qwen3-ASR-1.7B — Apple Neural Engine (fp16)

[![GitHub](https://img.shields.io/badge/GitHub-ane--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/ane-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![MLX build](https://img.shields.io/badge/MLX-bf16-1f6feb)](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx)

Core AI conversion of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) that runs entirely on the Apple Neural Engine, so transcription stays quick while the GPU is busy with something else. It runs through the [`ane-speech`](https://github.com/appautomaton/ane-speech) Swift runtime, with no Python, PyTorch, or cloud API at inference time.

## Model Details

- Developed by: [App Automaton](https://appautomaton.com)
- Upstream model: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (code: [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR))
- Conversion: from the official bf16 weights, by tracing the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) MLX graph and rebuilding it with Core AI; no PyTorch involved
- Precision: fp16 weights and activations. Activation ranges were measured before conversion and stay well inside fp16.
- Placement: every layer runs on the Neural Engine
- Parity: transcripts match the MLX bf16 build word for word on our test clips, apart from rare near-ties
- Input: 16 kHz mono audio, up to about a minute per call
- Requirements: a Mac with Apple silicon and macOS 27 or later

## Files

| Path | Contents |
| --- | --- |
| `audio_encoder/` | the audio encoder, identical to the one in [Confucius4-R2T2](https://huggingface.co/appautomaton/confucius4-r2t2-fp16-ane) |
| `qwen3_asr_1_7b/` | the text decoder in four segments and an output head; `runtime/` holds the embedding table, tokenizer, and audio front-end constants |

## How to Get Started

Install [`ane-speech`](https://pypi.org/project/ane-speech/); the model downloads from this repo on first use.

```bash
uv tool install ane-speech
ane-speech transcribe speech.wav --checkpoint qwen3_asr_1_7b
ane-speech transcribe speech.wav --checkpoint qwen3_asr_1_7b \
  --language Chinese --context "names and terms to expect"
```

From Python (`pip install ane-speech`):

```python
import ane_speech

asr = ane_speech.load("qwen3_asr_1_7b")
result = asr.generate("speech.wav")      # a file, or 16 kHz mono float32 samples
print(result.language, result.text)
```

The first load compiles the model for your Mac's Neural Engine and caches the result, so later loads take a fraction of a second. `ane-speech pull qwen3_asr_1_7b` downloads ahead of time.

## Notes

- One recording at a time. For text that appears while you speak, use [Confucius4-R2T2](https://huggingface.co/appautomaton/confucius4-r2t2-fp16-ane).
- Each call takes up to about a minute of audio; split longer recordings.
- This repo contains Core AI models only, with no PyTorch or safetensors checkpoint.

## Links

- Source code: [`appautomaton/ane-speech`](https://github.com/appautomaton/ane-speech)
- MLX build: [`appautomaton/qwen3-asr-1.7b-bf16-mlx`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx)
- Streaming model: [`appautomaton/confucius4-r2t2-fp16-ane`](https://huggingface.co/appautomaton/confucius4-r2t2-fp16-ane)
- More from App Automaton: [Project](https://appautomaton.com) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

Apache 2.0, following the upstream [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) release; a copy is included as [`LICENSE`](LICENSE). The `ane-speech` runtime is licensed separately under MIT.
