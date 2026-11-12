from pathlib import Path

import pytest

import ane_speech.hub as hub


def test_pull_fetches_the_encoder_and_one_checkpoint(monkeypatch):
    calls = []

    def snapshot_download(repo, allow_patterns):
        calls.append((repo, allow_patterns))
        return "/cache/snapshot"

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    assert hub.pull("qwen3_asr_1_7b") == Path("/cache/snapshot/qwen3_asr_1_7b")
    assert calls == [("appautomaton/qwen3-asr-1.7b-fp16-ane", ["audio_encoder/*", "qwen3_asr_1_7b/*"])]


def test_pull_names_the_checkpoints_it_knows():
    with pytest.raises(ValueError, match="confucius4_r2t2, qwen3_asr_1_7b"):
        hub.pull("whisper")
