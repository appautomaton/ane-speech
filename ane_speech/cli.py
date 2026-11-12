"""`ane-speech`, as installed from PyPI: the Swift CLI, with models found by name.

`pull` downloads checkpoints. `transcribe`, `stream`, and `serve` run the one
named by `--checkpoint`, downloading it first if needed, unless `--artifacts`
names a folder. Everything else reaches the Swift CLI unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ane_speech.hub import CHECKPOINTS, DEFAULT, pull
from ane_speech.model import binary

RUNS_A_MODEL = {"transcribe", "stream", "serve"}
PULL_USAGE = f"""\
       ane-speech pull [<checkpoint> ...]
           downloads models ({', '.join(CHECKPOINTS)}); they also download on first use
"""


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else None
    if command in (None, "-h", "--help", "help"):
        code = subprocess.run([binary()]).returncode  # prints the Swift CLI's usage
        sys.stderr.write(PULL_USAGE)
        sys.exit(code)
    if command == "--version":
        from ane_speech import __version__

        print(__version__)
        return
    if command == "pull":
        for checkpoint in args[1:] or [DEFAULT]:
            print(pull(checkpoint))
        return
    if command in RUNS_A_MODEL and "--artifacts" not in args:
        args = [*args, "--artifacts", str(pull(_option(args, "--checkpoint") or DEFAULT))]
    os.execv(binary(), [str(binary()), *args])


def _option(args: list[str], name: str) -> str | None:
    i = args.index(name) if name in args else -1
    return args[i + 1] if 0 <= i < len(args) - 1 else None
