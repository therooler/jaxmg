import pytest

# tests/test_readme_subprocess.py
import subprocess
import  sys
import  pathlib
import jax

devices = [d for d in jax.devices() if d.platform == "gpu"]
ndev = len(devices)

@pytest.mark.skipif(ndev not in (1, 2),
                    reason="readme.py expects 1 or 2 GPUs")
def test_readme_script():
    script = pathlib.Path(__file__).resolve().parents[2] / "examples" / "readme.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

@pytest.mark.skipif(ndev not in (1, 2),
                    reason="readme.py expects 1 or 2 GPUs")
def test_potrs_no_shardmap_script():
    script = pathlib.Path(__file__).resolve().parents[2] / "examples" / "potrs_no_shardmap.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

@pytest.mark.skipif(ndev not in (1, 2),
                    reason="block_cyclic_example_1.py expects 1 or 2 GPUs")
def test_block_cyclic_example_1_script():
    script = pathlib.Path(__file__).resolve().parents[2] / "examples" / "block_cyclic_example_1.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
