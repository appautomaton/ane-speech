# ane-speech

Speech models on the Apple Neural Engine (ANE) through Core AI. Sibling of
`../mlx-speech`, whose code traces the models and whose bf16 checkpoints are
the main weight source.

Converted so far, one family (`qwen3_asr`) with two checkpoints of one
architecture:

| Checkpoint | Converted from | Used as |
| --- | --- | --- |
| `confucius4_r2t2` | mlx-speech `models/netease/confucius4_r2t2/mlx-bf16` | streaming (R2T2) |
| `qwen3_asr_1_7b` | official bf16 release, `~/dev/ai/models/speech/ASR/Qwen3-ASR-1.7B` | one take at a time |

R2T2 finetuned only the decoder (and the embedding table), so the two share
the audio encoder bit for bit. The artifacts are self-contained:

```
artifacts/qwen3_asr/
  audio_encoder/     the shared encoder; tracing either checkpoint checks it matches
  <checkpoint>/      decoder models (functions `segment`, T=128, and `step`, T=1),
                     and runtime/: constants, embeddings.f16, tokenizer/
```

Two halves, one direction: Python converts once, Swift runs.

```
bf16 safetensors (mlx-speech models/ or ~/dev/ai/models)
  -> reference/    trace mlx-speech's MLX graph (mx.export_function)
  -> convert/      translate MLX primitives to coreai.authoring ops, fp16
  -> artifacts/    .aimodel, plus the embedding table and tokenizer the runtime needs
  -> Sources/      Swift runtime: import CoreAI, prefer the Neural Engine; reads only artifacts/
```

## Hard rules

- **No litter.** Every file a run creates lives in this repo:
  - converted models in `artifacts/`
  - SwiftPM output in `.build/`
  - scratch in `.tmp/`

  Importing `convert` points `TMPDIR` at `.tmp/<pid>/` and deletes that
  directory at exit. Core AI keeps compiled models in a system cache that
  cannot be relocated; `ane-speech evict` removes a model's entries. The first
  time a new tool or model runs, record what appeared outside the repo under
  "Footprint" below.
- **Checkpoints feed conversion only.** Conversion reads checkpoints in place
  from mlx-speech `models/` or this machine's weights folder (`~/dev/ai/models`,
  by category) and never copies them whole; `ANE_SPEECH_MLX_MODELS` and
  `ANE_SPEECH_WEIGHTS` override the two locations, and
  `convert/qwen3_asr/__init__.py` lists the checkpoints. What the runtime needs
  is exported into `artifacts/`, and the runtime reads nothing else.
- **Placement is a preference.** `SpecializationOptions` can prefer a compute
  unit but cannot pin one, and Core AI silently moves the whole graph to the GPU
  when the ANE compiler rejects any part of it. Placement is settled when a
  model is first compiled on an OS build, so check it after every conversion
  and every macOS update ("Placement check" below).
- **Parity before use.** Check every converted graph against the mlx-speech
  output for the same input before anything depends on it.
- **Assume fp16 on the ANE.** Before converting a block, measure its activation
  range in MLX. A block that overflows fp16 stays off the ANE.
- **No torch.** Conversion goes MLX trace -> `coreai.authoring`. The Swift
  runtime never imports or shells out to Python.
- **No aggressive performance testing.** `docs/performance.md` is a one-time
  record. Rerun its commands only when a change needs a before-and-after
  comparison, and add the result there instead of building a suite.

## Layout

```
convert/                 # Python: one subpackage per family
  workspace.py           #   repo paths, TMPDIR confinement
  <family>/              #   MLX graph translation, export entry point
  tests/
reference/               # Python under mlx-speech's env: MLX traces and reference outputs
Sources/
  ANESpeech/             # Swift library
    Runtime/             #   load .aimodel, compute-unit preference, cache eviction
    Audio/ Text/         #   shared front end and tokenizer, added with the first family
    <Family>/            #   one folder per family pipeline
  ane-speech/            # CLI
Tests/ANESpeechTests/    # Swift tests, added with the first testable unit
docs/performance.md      # one-time performance record; read it before measuring anything
artifacts/               # generated, gitignored; all the runtime reads, rebuilt only by conversion
.build/ .tmp/            # generated, gitignored, safe to delete
```

Python tests live in `convert/tests/` because APFS is case-insensitive and
`tests/` would collide with SwiftPM's `Tests/`.

A family is a module on each side: `convert/<family>/` produces its artifacts and
`Sources/ANESpeech/<Family>/` runs them. Shared code moves to `workspace.py`,
`Runtime/`, `Audio/`, or `Text/` only once a second family needs it.

## Commands

```sh
uv sync                                         # conversion env in .venv
uv run python -m convert.<family>               # export to artifacts/ (qwen3_asr: --checkpoint <name>)
uv run pytest                                   # convert/tests
uv run --project ../mlx-speech python -m reference.<script>  # needs Metal: unsandboxed; --checkpoint <name>
uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_trace --cache <rows>  # default 1152; docs/performance.md
swift build
swift run ane-speech inspect artifacts/<model>.aimodel
swift run ane-speech evict artifacts/<model>.aimodel
.build/release/ane-speech check [--checkpoint <name>]  # offline parity vs mlx-speech (swift build -c release)
.build/release/ane-speech stream-check   # R2T2 streaming parity vs artifacts/qwen3_asr/confucius4_r2t2/runtime/stream_*.json
.build/release/ane-speech stream <audio> [--chunk-ms 640]
.build/release/ane-speech transcribe <audio> [--checkpoint <name>] [--context <text>]
.build/release/ane-speech serve [--checkpoint <name>]  # resident helper, JSON lines on stdin/stdout (tnt-asr)
```

### Converting a checkpoint

In order, for a checkpoint named in `convert/qwen3_asr/__init__.py`. MLX and
Core AI need Metal and the ANE, so run these outside any sandbox.

```sh
uv sync
uv run --project ../mlx-speech python -m reference.qwen3_asr_encoder_ranges --checkpoint <name>   # fp16 headroom, first time only
uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_ranges --checkpoint <name>
uv run --project ../mlx-speech python -m reference.qwen3_asr_encoder_trace --checkpoint <name>    # shared encoder: write, or confirm a match
uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_trace --checkpoint <name>    # [--cache <rows>]
uv run --project ../mlx-speech python -m reference.qwen3_asr_runtime_assets --checkpoint <name>  # constants, embeddings, tokenizer
uv run python -m convert.qwen3_asr --checkpoint <name>    # build and check against MLX; placement
swift build -c release && .build/release/ane-speech check --checkpoint <name>
```

Before overwriting a model a running process has loaded, evict its cache
entries or set the old files aside and evict them once nothing holds them
(`ane-speech evict` needs the old file). The converter's own checks leave a
copy in the `python3/` cache, which has no evict API; delete it by hand.

## Placement check

The first load of a model on an OS build compiles it with that build's ANE
compiler and caches the result in
`~/Library/Caches/coreai-cache/<OS build>/<process name>/`. Later loads reuse the
cached compile, so placement only changes when the model is reconverted or
macOS updates. The runtime never checks placement; an agent checks it by hand
at those two moments:

1. Load the models once: `.build/release/ane-speech check --checkpoint <name>`
   for each checkpoint, or use both models in tnt-asr.
2. Read each entry's compile record:

   ```sh
   for m in ~/Library/Caches/coreai-cache/$(sw_vers -buildVersion)/ane-speech/*/*/model.aimodelx/*-this-delegates/MPSGraph/*/manifest.plist; do
     plutil -p "$m" | grep -q mps.fullyPlacedOnANE && echo "ANE $m" || echo "GPU $m"
   done
   ```

   Expect the shared encoder once, then four decoder segments and the head
   per checkpoint (eleven entries for two), all ANE, each flagged
   `mps.noGPUActivity`.
3. While dictating, watch `mactop`: ANE activity rises and the GPU stays at its
   background level.

The manifest is an undocumented Core AI file. If an OS update changes its
format, rely on step 3 and update step 2.

## Dependencies

Add one only when the implementation proves it necessary.

| Package | Stance |
| --- | --- |
| `CoreAI` (system framework) | Swift runtime; no Swift packages yet |
| `coreai-core`, `numpy` | Conversion only: `coreai.authoring` builds graphs, `coreai.runtime` checks them |
| `coreai-opt` | Not yet. It builds native extensions; measure its footprint first |

## Footprint

What runs leave outside the repo. Measured, not assumed.

| Source | Location | Size | Cleanup |
| --- | --- | --- | --- |
| `swift build` | `$(getconf DARWIN_USER_CACHE_DIR)clang/ModuleCache` | ~0.7 MB, shared by all Swift builds | clang prunes entries unused for 4 days |
| `swift build` | `~/Library/Caches/org.swift.swiftpm/manifests` | one row in a shared manifest DB | `swift package purge-cache` |
| `uv sync` | `~/.cache/uv/archive-v0` | ~580 MB of unpacked wheels, torch ~490 MB; 15 of 29 packages were already cached | `uv cache clean <package>`; `.venv` holds APFS clones and survives |
| `uv sync` | `$TMPDIR/uv-<hash>.lock` | empty lock file, one per project | OS temp sweep |
| MLX (`reference/`) | `$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal` | ~0.2 MB of compiled shaders, shared by unbundled Metal processes | system-managed |
| Core AI model load | `~/Library/Caches/coreai-cache/<OS build>/<process name>/` | one specialization per model and preference, about its weight size (24 MB for one encoder layer) | `ane-speech evict` for the CLI's entries; Python's land in `python3/`, which has no evict API |
| Core AI on the ANE (MPSGraph) | `$(getconf DARWIN_USER_TEMP_DIR)com.apple.MetalPerformanceShadersGraph/` | an empty `mpsgraph-<pid>-…` folder per ANE compile; ignores `TMPDIR` | `find … -type d -empty -delete`; OS temp sweep |
