"""Export what the Swift aligner reads, plus reference alignments to check it.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_aligner_assets

Writes to artifacts/qwen3_asr/qwen3_forced_aligner_0_6b/runtime/:

  runtime.json      front-end and model constants, the special token ids around
                    the audio, and the timestamp token and its class width
  mel_filters.f32   mlx-speech's mel filter bank, [n_fft / 2 + 1, n_mels]
  embeddings.f16    the input embedding table, [vocab, hidden], in fp16
  tokenizer/        the checkpoint's tokenizer files, as they are
  reference.json    per clip: the transcript and language, mlx-speech's text
                    units, prompt ids, and its float32 time classes and spans
                    (bf16 moves near-ties; see reference/qwen3_aligner_trace.py)

The prompt is <|audio_start|>, the audio placeholders, <|audio_end|>, then each
unit followed by two <timestamp> tokens. Added tokens split the string before
anything else, so each unit is tokenized alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx_speech.generation.qwen3_forced_aligner import Qwen3ForcedAligner
from mlx_speech.models.qwen3_asr.feature_extraction import _get_mel_filters
from mlx_speech.models.qwen3_asr.forced_aligner import MAX_ALIGN_SECONDS

from convert.qwen3_asr import ENCODER, align_clips, artifacts_dir, model_dir
from reference.qwen3_asr_runtime_assets import export_lookup_assets

CHECKPOINT = "qwen3_forced_aligner_0_6b"


def main() -> None:
    checkpoint, out_dir = model_dir(CHECKPOINT), artifacts_dir(CHECKPOINT) / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    aligner = Qwen3ForcedAligner.from_dir(checkpoint)
    exact = Qwen3ForcedAligner.from_dir(checkpoint, dtype=mx.float32)
    extractor, tower, tokenizer = aligner.feature_extractor, aligner.model.audio_tower, aligner.tokenizer
    text, settings = aligner.config.text_config, aligner.settings

    runtime = {
        # Provenance only; the runtime never reads it.
        "source": str(checkpoint.resolve()).replace(str(Path.home()), "~", 1),
        "encoder": ENCODER,  # relative to the checkpoint's artifacts: the aligner keeps its own
        "sample_rate": extractor.sample_rate,
        "n_fft": extractor.n_fft,
        "hop_length": extractor.hop_length,
        "n_mels": extractor.n_mels,
        "chunk_frames": tower.n_window * 2,
        "block_frames": tower.n_window_infer,
        "tokens_per_chunk": int(tower.output_lengths(tower.n_window * 2)),
        "hidden_size": text.hidden_size,
        "head_dim": text.head_dim,
        "rope_theta": text.rope_theta,
        "audio_token_id": tokenizer.token_to_id(tokenizer.audio_token),
        "audio_start_token_id": tokenizer.token_to_id(tokenizer.audio_bos_token),
        "audio_end_token_id": tokenizer.token_to_id(tokenizer.audio_eos_token),
        "timestamp_token_id": settings.timestamp_token_id,
        "timestamp_ms": settings.timestamp_segment_time_ms,
        "classes": settings.classify_num,
        "max_seconds": MAX_ALIGN_SECONDS,
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=1, ensure_ascii=False))
    filters = _get_mel_filters(sample_rate=extractor.sample_rate, n_fft=extractor.n_fft, n_mels=extractor.n_mels)
    filters.astype("<f4").tofile(out_dir / "mel_filters.f32")
    export_lookup_assets(aligner, checkpoint, out_dir)

    reference = {}
    for wav, transcript, language in align_clips():
        prepared = aligner.prepare(wav, transcript, language=language)
        result = exact.align(wav, transcript, language=language)
        reference[wav.stem] = {
            "wav": str(wav.resolve()),
            "text": transcript,
            "language": language,
            "units": list(prepared.units),
            "input_ids": list(prepared.input_ids),
            "classes": list(result.timestamp_classes),
            "spans": [list(span) for span in result.spans],
        }
        print(f"{wav.name}: {len(prepared.units)} units, {len(prepared.input_ids)} prompt ids")
    (out_dir / "reference.json").write_text(json.dumps(reference, indent=1, ensure_ascii=False))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
