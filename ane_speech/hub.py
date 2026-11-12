"""The converted checkpoints on Hugging Face, and where they land once downloaded.

Each repo mirrors artifacts/qwen3_asr/ in a source checkout: a recognizer's
holds the shared `audio_encoder/` beside its `<checkpoint>/` folder, which
finds the encoder at `../audio_encoder`; the aligner's holds only its folder,
its own encoder inside. Downloads go to the Hugging Face cache, so a model is
fetched once per machine and found again by name.

Each release pins the repo commits it was tested with, so uploading a new
model never changes what an installed version runs; a new model ships with a
new release.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINTS = {  # name: (repo, commit)
    "confucius4_r2t2": ("appautomaton/confucius4-r2t2-fp16-ane", "3a20edb1b5c56f1ed6c9e08b8aaefe1e5ed93082"),
    "qwen3_asr_1_7b": ("appautomaton/qwen3-asr-1.7b-fp16-ane", "f4621c3f10b108ea52baed35a8ee8c6c4b3406dd"),
    "qwen3_forced_aligner_0_6b": (
        "appautomaton/qwen3-forced-aligner-0.6b-fp16-ane", "bf46b4d5124dbcde28b46443e6f0848c878b03b1",
    ),
}
DEFAULT = "confucius4_r2t2"  # streams while you speak; qwen3_asr_1_7b takes one whole recording
ALIGNER = "qwen3_forced_aligner_0_6b"  # times a known transcript; it holds its own encoder
ENCODER = "audio_encoder"


def pull(checkpoint: str = DEFAULT) -> Path:
    """Downloads `checkpoint` unless the cache already holds it, and returns its
    folder. Without a network connection, a cached copy is used as is."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}; expected one of {', '.join(CHECKPOINTS)}")
    repo, revision = CHECKPOINTS[checkpoint]
    snapshot = snapshot_download(repo, revision=revision, allow_patterns=[f"{ENCODER}/*", f"{checkpoint}/*"])
    return Path(snapshot) / checkpoint
