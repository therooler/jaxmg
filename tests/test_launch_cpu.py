# test_launch_parallel.py

import subprocess
import sys
import os
from pathlib import Path


DEV_SETTER = Path(__file__).parent / "cpu_runner.py"

def launch_runner(ndev):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        ["env", sys.executable, "-u", str(DEV_SETTER), str(ndev)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_launch_multiple_configs_cpu():
    # Launch three processes with different ndev settings
    procs = [launch_runner(ndev) for ndev in [1, 2, 4]]

    # Wait and collect results
    for proc, ndev in zip(procs, [1, 2, 4]):
        stdout, stderr = proc.communicate()
        print(f"\n--- Output for ndev={ndev} ---")
        print(stdout)
        if stderr:
            print(f"[stderr for ndev={ndev}]:\n{stderr}")
        assert proc.returncode == 0, f"Process for ndev={ndev} failed!"
