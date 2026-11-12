"""Load converted models and check them: placement, parity, ANE activity."""

from __future__ import annotations

import json
import plistlib
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel, ComputeUnitKind, InferenceFunction, SpecializationOptions

CACHE = Path.home() / "Library/Caches/coreai-cache"


def _entries() -> set[Path]:
    return {p.parent.parent for p in CACHE.glob("*/*/*/*/model.aimodelx/*-this.odix")}


def _placement(entry: Path) -> str:
    manifests = list(entry.glob("model.aimodelx/*-this-delegates/MPSGraph/*/manifest.plist"))
    if not manifests:
        return "no MPSGraph package"
    text = repr(plistlib.loads(manifests[0].read_bytes()))
    if "mps.fullyPlacedOnANE" in text:
        return "fully on the ANE"
    regions = len(set(re.findall(r"_ANE_region_\d+_\d+", text)))
    return f"NOT fully on the ANE ({regions} ANE regions)"


async def load(path: Path, function: str) -> tuple[InferenceFunction, str]:
    """Load preferring the ANE; report where MPSGraph placed it if this load compiled it."""
    before = _entries()
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
    start = time.perf_counter()
    model = await AIModel.load(path, options)
    new = _entries() - before
    where = _placement(new.pop()) if new else "already compiled"
    note = f"{path.name}: {time.perf_counter() - start:.1f} s to load, {where}"
    return model.load_function(function), note


def rel_l2(y: np.ndarray, ref: np.ndarray) -> float:
    y, ref = y.astype(np.float64), ref.astype(np.float64)
    return float(np.linalg.norm(y - ref) / np.linalg.norm(ref))


def ane_activity(seconds: float, work: Callable[[float], None]) -> tuple[float, float]:
    """Mean ANE activity (%) from mactop: idle first, then while `work` runs."""
    proc = subprocess.Popen(
        ["mactop", "--headless", "--count", "0", "-i", "250"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    samples: list[tuple[float, float]] = []

    def read() -> None:
        for line in proc.stdout:
            line = line.strip().lstrip("[").rstrip(",]")
            if line:
                samples.append((time.perf_counter(), json.loads(line)["soc_metrics"]["ane_active"]))

    threading.Thread(target=read, daemon=True).start()
    t0 = time.perf_counter()
    time.sleep(seconds)
    t1 = time.perf_counter()
    work(seconds)
    t2 = time.perf_counter()
    time.sleep(0.4)
    proc.terminate()

    def mean(a: float, b: float) -> float:
        window = [v for t, v in samples if t - 0.25 >= a and t <= b]
        return sum(window) / len(window) if window else float("nan")

    return mean(t0, t1), mean(t1, t2)
