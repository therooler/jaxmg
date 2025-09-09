import pytest

# tests/test_readme_subprocess.py
import subprocess
import  sys
import  pathlib
import jax

@pytest.mark.skipif(len(jax.devices("gpu")) not in (1, 2),
                    reason="readme.py expects 1 or 2 GPUs")
def test_readme_script():
    script = pathlib.Path(__file__).resolve().parents[1] / "examples" / "readme.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
