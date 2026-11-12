import subprocess
import sys

from convert.workspace import ROOT, SCRATCH

PROBE = """
import tempfile
import convert
path = tempfile.mkstemp()[1]
print(tempfile.gettempdir())
print(path)
"""


def test_import_confines_temp_files_and_removes_them_at_exit():
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    tempdir, tempfile_path = out

    assert tempdir.startswith(str(SCRATCH))
    assert tempfile_path.startswith(tempdir)
    assert not (SCRATCH / tempdir.rsplit("/", 1)[1]).exists()
