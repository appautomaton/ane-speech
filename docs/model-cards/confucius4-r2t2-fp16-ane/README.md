---
license: other
license_name: netease-model-use-license-agreement
license_link: LICENSE
pipeline_tag: automatic-speech-recognition
base_model: netease-youdao/Confucius4-R2T2
language:
- zh
- en
- yue
- ar
- de
- fr
- es
- pt
- id
- it
- ko
- ru
- th
- vi
- ja
- tr
- hi
- ms
- nl
- sv
- da
- fi
- pl
- cs
- fil
- fa
- el
- ro
- hu
- mk
tags:
- core-ai
- apple-neural-engine
- apple-silicon
- asr
- speech-recognition
- streaming-asr
- real-time
- confucius4
- r2t2
---

# Confucius4-R2T2 — Apple Neural Engine (fp16)

[![GitHub](https://img.shields.io/badge/GitHub-ane--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/ane-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![MLX build](https://img.shields.io/badge/MLX-bf16-1f6feb)](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx)

Core AI conversion of NetEase Youdao's [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2), a streaming speech recognition model finetuned from Qwen3-ASR-1.7B. It runs entirely on the Apple Neural Engine, so text keeps up with your voice while the GPU is busy with something else, through the [`ane-speech`](https://github.com/appautomaton/ane-speech) Swift runtime with no Python, PyTorch, vLLM, or cloud API at inference time.

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

## Model Details

- Upstream developer: NetEase Youdao ([`netease-youdao/Confucius4-R2T2`](https://github.com/netease-youdao/Confucius4-R2T2))
- Core AI conversion and runtime: [App Automaton](https://appautomaton.com)
- Architecture: Qwen3-ASR graph (audio encoder + 1.7B text decoder)
- Conversion: from the [MLX bf16 build](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx), by tracing the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) MLX graph and rebuilding it with Core AI; no PyTorch involved
- Precision: fp16 weights and activations. Activation ranges were measured before conversion and stay well inside fp16.
- Placement: every layer runs on the Neural Engine
- Parity: streaming output matches `mlx-speech` window by window on our test clips
- Input: 16 kHz mono audio, up to about a minute per take
- Languages: 30, as declared by upstream
- Requirements: a Mac with Apple silicon and macOS 27 or later

## Files

| Path | Contents |
| --- | --- |
| `audio_encoder/` | the audio encoder, identical to the one in [Qwen3-ASR-1.7B](https://huggingface.co/appautomaton/qwen3-asr-1.7b-fp16-ane), which R2T2 left unchanged |
| `confucius4_r2t2/` | the text decoder in four segments and an output head; `runtime/` holds the embedding table, tokenizer, and audio front-end constants |

## How to Get Started

Build the runtime and download the model into it:

```bash
git clone https://github.com/appautomaton/ane-speech.git
cd ane-speech
swift build -c release
hf download appautomaton/confucius4-r2t2-fp16-ane --local-dir artifacts/qwen3_asr \
  --include "audio_encoder/*" --include "confucius4_r2t2/*"
```

Play a recording through the model as if it were spoken live, printing text as it settles, or transcribe it whole:

```bash
.build/release/ane-speech stream speech.wav --chunk-ms 480
.build/release/ane-speech transcribe speech.wav
```

The first load compiles the model for your Mac's Neural Engine and caches the result, so later loads take a fraction of a second. Apps can keep the model loaded with `ane-speech serve`, which answers JSON requests on stdin, live microphone audio included.

## Streaming

`ane-speech` implements NetEase's chunk loop as `mlx-speech` does: prefix rollback, a per-window token budget, and append-only commits. Each window reuses the closed audio-encoder blocks and the decoder's cached prefix, and recomputes only what changed.

## Notes

- One take holds up to about a minute of speech; start a new take for longer dictation.
- This repo contains Core AI models only, with no PyTorch or safetensors checkpoint.

## Links

- Source code: [`appautomaton/ane-speech`](https://github.com/appautomaton/ane-speech)
- MLX build: [`appautomaton/confucius4-r2t2-bf16-mlx`](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx)
- Offline model: [`appautomaton/qwen3-asr-1.7b-fp16-ane`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-fp16-ane)
- More from App Automaton: [Project](https://appautomaton.com) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

The model weights are licensed under the **NetEase Youdao Model Use License Agreement**; a copy is included in this repository as [`LICENSE`](LICENSE) (original: [MODEL_LICENSE](https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/refs/heads/master/MODEL_LICENSE)). By using these weights you agree to its terms, including the restrictions on large-scale commercial use (section 2.2) and prohibited high-risk use (section 4.2). Downstream redistribution must retain the license and this notice.

The `ane-speech` runtime is licensed separately under MIT.
