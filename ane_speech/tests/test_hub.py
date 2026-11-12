from pathlib import Path

import pytest

import ane_speech.hub as hub


def test_pull_fetches_the_encoder_and_one_checkpoint_at_the_pinned_commit(monkeypatch):
    calls = []

    def snapshot_download(repo, revision, allow_patterns):
        calls.append((repo, revision, allow_patterns))
        return "/cache/snapshot"

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    assert hub.pull("qwen3_asr_1_7b") == Path("/cache/snapshot/qwen3_asr_1_7b")
    repo, revision = hub.CHECKPOINTS["qwen3_asr_1_7b"]
    assert calls == [(repo, revision, ["audio_encoder/*", "qwen3_asr_1_7b/*"])]


def test_every_checkpoint_pins_a_full_commit():
    for repo, revision in hub.CHECKPOINTS.values():
        assert repo.startswith("appautomaton/") and len(revision) == 40, repo


def test_pull_names_the_checkpoints_it_knows():
    with pytest.raises(ValueError, match="confucius4_r2t2, qwen3_asr_1_7b, qwen3_forced_aligner_0_6b"):
        hub.pull("whisper")
