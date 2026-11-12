"""A converted checkpoint, loaded once in a resident `ane-speech serve` process.

The Swift CLI runs the models on the Neural Engine; this side speaks its JSON
lines protocol (Sources/ane-speech/Serve.swift) and nothing else. Method names
follow mlx-speech's ASR models, so code written against one runs on the other.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ane_speech.hub import DEFAULT, pull

SAMPLE_RATE = 16000
# Package.swift's platform. The wheel's tag can only claim an older macOS
# (hatch_build.py), so binary() checks this one.
MIN_MACOS = 27
# The first load on a new macOS build compiles the models for the Neural
# Engine, which takes far longer than the usual sub-second load.
_READY_TIMEOUT_SECONDS = 600


def binary() -> Path:
    """The ane-speech CLI: the copy inside the wheel, or a source checkout's release build."""
    release = platform.mac_ver()[0]
    if not release or int(release.split(".")[0]) < MIN_MACOS:
        raise RuntimeError(f"ane-speech needs macOS {MIN_MACOS} or later on Apple silicon; "
                           f"this is {f'macOS {release}' if release else platform.system()}")
    here = Path(__file__).resolve().parent
    for path in (here / "bin" / "ane-speech", here.parent / ".build" / "release" / "ane-speech"):
        if path.is_file():
            return path
    raise FileNotFoundError("ane-speech: the CLI is missing; in a source checkout, run `swift build -c release`")


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


@dataclass(frozen=True)
class StreamUpdate:
    committed: str
    """All text committed so far. It only grows; committed text is never revised."""
    hypothesis: str
    """The model's latest guess at the whole take."""
    language: str
    full: bool
    """The take has filled the model's cache and takes no more audio; finalize it."""


def load(checkpoint: str = DEFAULT, *, artifacts: str | os.PathLike[str] | None = None) -> Model:
    """Loads `checkpoint`, downloading it first if needed. `artifacts` names a
    converted checkpoint's folder instead, such as one under a source
    checkout's artifacts/qwen3_asr/."""
    return Model(Path(artifacts) if artifacts is not None else pull(checkpoint))


class Model:
    """One checkpoint in its own `ane-speech serve` process, which exits with
    this one. Calls are serialized; a helper that dies is restarted on the next
    call."""

    def __init__(self, artifacts: Path) -> None:
        self.artifacts = Path(artifacts).resolve()
        if not (self.artifacts / "decoder.json").is_file():
            raise FileNotFoundError(f"ane-speech: no converted checkpoint in {self.artifacts}")
        self._binary = binary()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._spawn()

    def generate(
        self,
        audio: str | os.PathLike[str] | np.ndarray,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        context: str = "",
    ) -> Transcript:
        """Transcribes a whole recording: an audio file, or 16 kHz mono samples.
        `language` forces the output language, which is otherwise detected;
        `context` is background for the model, such as names to expect."""
        source = {"path": os.path.abspath(audio)} if isinstance(audio, (str, os.PathLike)) else {"pcm": _pcm(audio, sample_rate)}
        reply = self._request({"op": "transcribe", **source, "language": language, "context": context})
        return Transcript(text=reply["text"], language=reply["language"])

    def stream_session(
        self,
        *,
        language: str | None = None,
        context: str = "",
        chunk_ms: int = 160,
        lookahead_ms: int = 160,
        unfixed_token_num: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> StreamSession:
        """Starts a live take, replacing any open one. Built for Confucius4-R2T2,
        the streaming checkpoint."""
        _check_rate(sample_rate)
        self._request({
            "op": "start", "language": language, "context": context, "chunk_ms": chunk_ms,
            "lookahead_ms": lookahead_ms, "unfixed_tokens": unfixed_token_num,
        })
        return StreamSession(self)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _spawn(self) -> None:
        proc = subprocess.Popen(
            [str(self._binary), "serve", "--artifacts", str(self.artifacts)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        ready = threading.Event()
        errors: list[str] = []

        def drain() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").strip()
                if line == "ANE_READY":
                    ready.set()
                elif line:
                    errors.append(line)
            ready.set()  # EOF: the helper exited

        threading.Thread(target=drain, name="ane-speech-stderr", daemon=True).start()
        if not ready.wait(_READY_TIMEOUT_SECONDS) or proc.poll() is not None:
            proc.kill()
            raise RuntimeError(f"ane-speech failed to start: {errors[-1] if errors else 'no output'}")
        self._proc = proc

    def _request(self, payload: dict) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            proc = self._proc
            assert proc is not None and proc.stdin is not None and proc.stdout is not None
            proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            proc.stdin.flush()
            reply = proc.stdout.readline()
        if not reply:
            raise RuntimeError("ane-speech exited mid-request")
        data = json.loads(reply)
        if "error" in data:
            raise RuntimeError(f"ane-speech: {data['error']}")
        return data


class StreamSession:
    """One live take: `feed` audio as it arrives, then `finalize`."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def feed(self, pcm: np.ndarray) -> StreamUpdate:
        """Adds 16 kHz mono samples, any number at a time."""
        reply = self._model._request({"op": "feed", "pcm": _pcm(pcm, SAMPLE_RATE)})
        return StreamUpdate(committed=reply["committed"], hypothesis=reply["hypothesis"],
                            language=reply["language"], full=reply["full"])

    def finalize(self) -> Transcript:
        """Ends the take and commits the rest of its text."""
        reply = self._model._request({"op": "finalize"})
        return Transcript(text=reply["text"], language=reply["language"])


def _check_rate(sample_rate: int) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"ane-speech takes {SAMPLE_RATE} Hz audio; got {sample_rate} Hz")


def _pcm(audio: np.ndarray, sample_rate: int) -> str:
    """Samples as the helper reads them: base64 of little-endian float32."""
    _check_rate(sample_rate)
    return base64.b64encode(np.ascontiguousarray(audio, dtype="<f4").reshape(-1).tobytes()).decode("ascii")
