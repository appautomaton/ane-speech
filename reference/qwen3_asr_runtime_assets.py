"""Export what the Swift Qwen3-ASR runtime reads, plus reference outputs to check it.

Runs under mlx-speech's environment from the repo root; MLX needs Metal, so run
it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_runtime_assets [wav ...]

Writes to artifacts/qwen3_asr/<checkpoint>/runtime/:

  runtime.json      front-end and model constants, special token ids, and the
                    prompt around the audio span as token ids (no context;
                    `language_suffix` forces a language)
  mel_filters.f32   mlx-speech's mel filter bank, [n_fft / 2 + 1, n_mels]
  reference.json    per clip: mlx-speech's offline transcript and tokens, and
                    the shape of its log-mel features
  <clip>.mel.f32    those features, [n_mels, frames]
  embeddings.f16    the input embedding table, [vocab, hidden], bf16 widened
                    to fp16 exactly as the runtime used to on every lookup
  tokenizer/        the checkpoint's tokenizer files, as they are

With these and the converted models, the runtime reads nothing outside
artifacts/; the checkpoint is needed only to convert.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_speech.generation.qwen3_asr import Qwen3ASRTranscriber
from mlx_speech.models.qwen3_asr.feature_extraction import _get_mel_filters

from convert.qwen3_asr import ENCODER, add_checkpoint_argument, artifacts_dir, model_dir, sample_wav
from convert.workspace import mlx_models_dir

CLIPS = [
    sample_wav(),
    mlx_models_dir() / "ibm/granite_4_0_1b_speech/mlx-int8/multilingual_sample.wav",
]


# The tokenizer files the runtime reads: tokenizer.json when the checkpoint has
# one, else the vocabulary, merges, and added tokens of Qwen2Tokenizer.
TOKENIZER_FILES = (("tokenizer.json",), ("vocab.json", "merges.txt", "tokenizer_config.json"))


def export_lookup_assets(asr: Qwen3ASRTranscriber, checkpoint: Path, out_dir: Path) -> None:
    """The embedding table and the tokenizer: what the runtime would otherwise
    read from the checkpoint."""
    table = np.array(asr.model.text_decoder.model.embed_tokens.weight.astype(mx.float32)).astype("<f2")
    if not np.isfinite(table).all():
        raise SystemExit("the embedding table overflows fp16")
    table.tofile(out_dir / "embeddings.f16")
    files = next(names for names in TOKENIZER_FILES if all((checkpoint / n).exists() for n in names))
    tokenizer = out_dir / "tokenizer"
    shutil.rmtree(tokenizer, ignore_errors=True)
    tokenizer.mkdir()
    for name in files:
        shutil.copyfile(checkpoint / name, tokenizer / name)
    print(f"embeddings: {table.shape[0]} x {table.shape[1]} fp16; tokenizer: {', '.join(files)}")


def split_prompt(ids: list[int], audio_id: int) -> tuple[list[int], list[int]]:
    at = ids.index(audio_id)
    assert ids.count(audio_id) == 1
    return ids[:at], ids[at + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", type=Path, nargs="*", default=CLIPS)
    add_checkpoint_argument(parser)
    args = parser.parse_args()
    checkpoint, out_dir = model_dir(args.checkpoint), artifacts_dir(args.checkpoint) / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = Qwen3ASRTranscriber.from_dir(checkpoint)
    processor, config = asr.processor, asr.config
    extractor, tower = processor.feature_extractor, asr.model.audio_tower
    audio_id = config.audio_token_id
    prefix, suffix = split_prompt(processor.build_prompt(context="", audio_length=1).input_ids, audio_id)
    languages = {}
    for language in processor.supported_languages:
        _, forced = split_prompt(processor.build_prompt(audio_length=1, language=language).input_ids, audio_id)
        assert forced[: len(suffix)] == suffix
        languages[language] = forced[len(suffix) :]

    text = config.text_config
    runtime = {
        # Provenance only; the runtime never reads it. "~" keeps the home folder out
        # of artifacts that are published.
        "source": str(checkpoint.resolve()).replace(str(Path.home()), "~", 1),
        "encoder": f"../{ENCODER}",  # relative to the checkpoint's artifacts
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
        "vocab_size": text.vocab_size,
        "audio_token_id": audio_id,
        "eos_token_ids": list(asr.eos_token_ids),
        "prompt_prefix": prefix,
        "prompt_suffix": suffix,
        "language_suffix": languages,
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=1, ensure_ascii=False))
    filters = _get_mel_filters(sample_rate=extractor.sample_rate, n_fft=extractor.n_fft, n_mels=extractor.n_mels)
    filters.astype("<f4").tofile(out_dir / "mel_filters.f32")
    export_lookup_assets(asr, checkpoint, out_dir)

    reference = {}
    for wav in args.wavs:
        batch = extractor(wav)
        frames = int(batch.feature_attention_mask.sum())
        mel = batch.input_features[0, :, :frames].astype("<f4")
        mel.tofile(out_dir / f"{wav.stem}.mel.f32")
        result = asr.transcribe(wav)
        reference[wav.stem] = {
            "wav": str(wav.resolve()),
            "mel_shape": list(mel.shape),
            "language": result.language,
            "text": result.text,
            "raw_text": result.raw_text,
            "tokens": result.tokens,
        }
        print(f"{wav.name}: {frames} frames, {len(result.tokens)} tokens: {result.text}")
    (out_dir / "reference.json").write_text(json.dumps(reference, indent=1, ensure_ascii=False))
    print(f"prompt: {len(prefix)} ids, audio span, {len(suffix)} ids; {len(languages)} languages -> {out_dir}")


if __name__ == "__main__":
    main()
