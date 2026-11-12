"""The converted checkpoints on Hugging Face, and where they land once downloaded.

Each repo mirrors artifacts/qwen3_asr/ in a source checkout: the shared
`audio_encoder/` beside one `<checkpoint>/` folder, which finds the encoder at
`../audio_encoder`. Downloads go to the Hugging Face cache, so a model is
fetched once per machine and found again by name.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "confucius4_r2t2": "appautomaton/confucius4-r2t2-fp16-ane",  # streams while you speak
    "qwen3_asr_1_7b": "appautomaton/qwen3-asr-1.7b-fp16-ane",  # one whole recording at a time
}
DEFAULT = "confucius4_r2t2"
ENCODER = "audio_encoder"


def pull(checkpoint: str = DEFAULT) -> Path:
    """Downloads `checkpoint` unless the cache already holds it, and returns its
    folder. Without a network connection, a cached copy is used as is."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}; expected one of {', '.join(CHECKPOINTS)}")
    snapshot = snapshot_download(CHECKPOINTS[checkpoint], allow_patterns=[f"{ENCODER}/*", f"{checkpoint}/*"])
    return Path(snapshot) / checkpoint
