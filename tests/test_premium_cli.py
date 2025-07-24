import subprocess
import sys


def test_premium_cli_runs():
    result = subprocess.run(
        [sys.executable, "interface/premium_cli.py", "--steps", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Demo complete" in result.stdout
