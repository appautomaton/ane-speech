"""The serve protocol, against a stand-in for the Swift helper."""

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import ane_speech.model
from ane_speech import Model, StreamUpdate, Transcript

FAKE_SERVE = """
import base64, json, sys
sys.stderr.write("loading\\nANE_READY\\n")
sys.stderr.flush()
for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    if op == "transcribe":
        source = request.get("path") or f"{len(base64.b64decode(request['pcm'])) // 4} samples"
        reply = {"text": source, "language": request["language"] or "English"}
    elif op == "start":
        reply = {"ok": True}
    elif op == "feed":
        samples = len(base64.b64decode(request["pcm"])) // 4
        reply = {"committed": "hi", "hypothesis": f"hi {samples}", "language": "English", "full": False}
    elif op == "finalize":
        reply = {"text": "hi there", "language": "English"}
    elif op == "argv":
        reply = {"argv": sys.argv[1:]}
    elif op == "exit":
        sys.exit(0)
    else:
        reply = {"error": f"unknown op {op}"}
    print(json.dumps(reply), flush=True)
"""


@pytest.fixture
def model(tmp_path, monkeypatch):
    serve = tmp_path / "ane-speech"
    serve.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SERVE))
    serve.chmod(0o755)
    monkeypatch.setattr(ane_speech.model, "binary", lambda: serve)
    artifacts = tmp_path / "qwen3_asr_1_7b"
    artifacts.mkdir()
    (artifacts / "decoder.json").write_text("{}")
    with Model(artifacts) as model:
        yield model


def test_starts_serve_on_the_artifacts(model):
    assert model._request({"op": "argv"})["argv"] == ["serve", "--artifacts", str(model.artifacts)]


def test_generate_sends_samples_or_an_absolute_path(model, monkeypatch, tmp_path):
    assert model.generate(np.zeros(1600)) == Transcript("1600 samples", "English")
    monkeypatch.chdir(tmp_path)
    assert model.generate("speech.wav", language="Chinese") == Transcript(str(tmp_path / "speech.wav"), "Chinese")


def test_stream_session(model):
    session = model.stream_session(chunk_ms=480)
    assert session.feed(np.zeros(320, dtype=np.float32)) == StreamUpdate("hi", "hi 320", "English", False)
    assert session.finalize() == Transcript("hi there", "English")


def test_rejects_other_sample_rates(model):
    with pytest.raises(ValueError, match="16000 Hz"):
        model.generate(np.zeros(10), sample_rate=44100)


def test_raises_helper_errors(model):
    with pytest.raises(RuntimeError, match="unknown op nope"):
        model._request({"op": "nope"})


def test_restarts_a_helper_that_exited(model):
    with pytest.raises(RuntimeError, match="exited"):
        model._request({"op": "exit"})
    assert model.generate(np.zeros(16)).text == "16 samples"


def test_refuses_a_folder_without_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="no converted checkpoint"):
        Model(tmp_path)


def test_min_macos_follows_package_swift():
    swift = (Path(__file__).parents[2] / "Package.swift").read_text()
    assert re.search(r'\.macOS\("(\d+)', swift)[1] == str(ane_speech.model.MIN_MACOS)
