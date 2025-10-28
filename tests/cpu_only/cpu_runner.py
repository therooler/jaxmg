import sys
import os
import pytest
ndev = int(sys.argv[1])
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
import jax
from pathlib import Path

# Setup JAX before anything else imports it
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

devices = jax.devices()
print(f"JAX initialized with {len(devices)} devices (requested {ndev})")
assert len(devices) == ndev

# Now run test modules 
exit_code = pytest.main(["-s", f"{Path(__file__).parent}/block_cyclic_cpu_test.py", "-v"])

sys.exit(exit_code)
