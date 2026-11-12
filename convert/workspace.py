"""Repository paths and scratch confinement for conversion runs.

torch.export, coreai-torch, and the compilers they call write scratch files to
the system temp directory. `confine()` points `TMPDIR` at a per-process folder
under `.tmp/` and deletes it at exit, so a run leaves nothing elsewhere on the
machine. It must run before torch is imported; `convert/__init__.py` calls it.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SCRATCH = ROOT / ".tmp"


def mlx_models_dir() -> Path:
    """mlx-speech's checkpoint folder. Weights are read in place, never copied."""
    default = ROOT.parent / "mlx-speech" / "models"
    return Path(os.environ.get("ANE_SPEECH_MLX_MODELS", default)).resolve()


def weights_dir() -> Path:
    """This machine's weights folder (~/dev/ai/models), for checkpoints mlx-speech
    does not hold. Weights are read in place, never copied."""
    default = ROOT.parent.parent / "models"
    return Path(os.environ.get("ANE_SPEECH_WEIGHTS", default)).resolve()


def confine() -> Path:
    """Send this process's temp files to `.tmp/<pid>/` and remove it at exit."""
    scratch = SCRATCH / str(os.getpid())
    if os.environ.get("TMPDIR") == str(scratch):
        return scratch
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    return scratch
