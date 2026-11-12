"""Builds the Swift CLI into the wheel, which makes the wheel macOS arm64 only,
for the macOS that Package.swift requires."""

import re
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SwiftCLI(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version == "editable":
            return  # an editable install runs the CLI from .build/release/
        root = Path(self.root)
        build = ["swift", "build", "-c", "release", "--product", "ane-speech"]
        subprocess.run(build, cwd=root, check=True)
        bin_dir = subprocess.run([*build, "--show-bin-path"], cwd=root, check=True, capture_output=True, text=True)
        macos = re.search(r'\.macOS\("(\d+)', (root / "Package.swift").read_text())
        if macos is None:
            raise RuntimeError("Package.swift names no macOS version")
        build_data["force_include"][str(Path(bin_dir.stdout.strip()) / "ane-speech")] = "ane_speech/bin/ane-speech"
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-macosx_{macos[1]}_0_arm64"
