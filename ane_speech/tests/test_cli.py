"""The PyPI entry point: which commands get a model, and from where."""

from pathlib import Path

import pytest

import ane_speech.cli as cli

BINARY = Path("/opt/ane-speech")


class Exec(Exception):
    pass


@pytest.fixture
def run(monkeypatch):
    pulled = []

    def pull(checkpoint):
        pulled.append(checkpoint)
        return Path("/cache") / checkpoint

    def execv(path, argv):
        raise Exec(argv)

    monkeypatch.setattr(cli, "pull", pull)
    monkeypatch.setattr(cli, "binary", lambda: BINARY)
    monkeypatch.setattr(cli.os, "execv", execv)

    def run(*args):
        try:
            cli.main(list(args))
        except Exec as exec_:
            return exec_.args[0][1:], pulled
        return None, pulled

    return run


def test_model_commands_get_the_named_checkpoint(run):
    argv, pulled = run("transcribe", "a.wav", "--checkpoint", "qwen3_asr_1_7b")
    assert argv == ["transcribe", "a.wav", "--checkpoint", "qwen3_asr_1_7b", "--artifacts", "/cache/qwen3_asr_1_7b"]
    assert pulled == ["qwen3_asr_1_7b"]


def test_model_commands_default_to_r2t2(run):
    assert run("stream", "a.wav")[0] == ["stream", "a.wav", "--artifacts", "/cache/confucius4_r2t2"]


def test_align_defaults_to_the_aligner(run):
    argv, pulled = run("align", "a.wav", "--text", "hi")
    assert argv == ["align", "a.wav", "--text", "hi", "--artifacts", "/cache/qwen3_forced_aligner_0_6b"]
    assert pulled == ["qwen3_forced_aligner_0_6b"]


def test_explicit_artifacts_skip_the_download(run):
    assert run("serve", "--artifacts", "here") == (["serve", "--artifacts", "here"], [])


def test_other_commands_pass_through(run):
    assert run("inspect", "m.aimodel") == (["inspect", "m.aimodel"], [])


def test_pull(run, capsys):
    assert run("pull", "qwen3_asr_1_7b", "confucius4_r2t2") == (None, ["qwen3_asr_1_7b", "confucius4_r2t2"])
    assert capsys.readouterr().out.split() == ["/cache/qwen3_asr_1_7b", "/cache/confucius4_r2t2"]


def test_version(run, capsys):
    from ane_speech import __version__

    assert run("--version") == (None, [])
    assert capsys.readouterr().out.strip() == __version__
