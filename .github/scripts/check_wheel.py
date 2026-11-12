"""Checks the built wheel before anything uploads it.

    python3 .github/scripts/check_wheel.py dist/*.whl

The wheel must be the macOS arm64 build with the Swift CLI inside, executable;
it must install into a clean environment and run there; and when a tag
triggered the run, the tag must name the wheel's version.
"""

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def fail(message: str) -> None:
    sys.exit(f"check_wheel: {message}")


def main(paths: list[str]) -> None:
    if len(paths) != 1:
        fail(f"expected one wheel, found {len(paths)}")
    wheel = Path(paths[0])
    name, version, python, abi, platform = wheel.stem.split("-")
    if (name, python, abi) != ("ane_speech", "py3", "none") or not re.fullmatch(r"macosx_\d+_0_arm64", platform):
        fail(f"unexpected wheel {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        info = archive.getinfo("ane_speech/bin/ane-speech")
        if (info.external_attr >> 16) & 0o111 != 0o111:
            fail("the CLI in the wheel is not executable")

    if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ["GITHUB_REF_NAME"] != f"v{version}":
        fail(f"tag {os.environ['GITHUB_REF_NAME']} does not match version {version}")

    with tempfile.TemporaryDirectory() as env:
        subprocess.run(["uv", "venv", "--quiet", env], check=True)
        subprocess.run(["uv", "pip", "install", "--quiet", "--python", f"{env}/bin/python", str(wheel)], check=True)
        cli = Path(env) / "bin" / "ane-speech"
        reported = subprocess.run([cli, "--version"], check=True, capture_output=True, text=True).stdout.strip()
        if reported != version:
            fail(f"installed CLI reports {reported}, expected {version}")
        usage = subprocess.run([cli], capture_output=True, text=True)
        if usage.returncode != 2 or "usage: ane-speech transcribe" not in usage.stderr:
            fail(f"the Swift CLI did not run:\n{usage.stderr}")

    print(f"check_wheel: {wheel.name} ok")


if __name__ == "__main__":
    main(sys.argv[1:])
