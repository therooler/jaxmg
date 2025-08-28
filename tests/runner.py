# run_test_with_devices.py
import sys
import os
import pytest
ndev = int(sys.argv[1])
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
import jax
# Setup JAX before anything else imports it
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

devices = jax.devices()
print(f"JAX initialized with {len(devices)} devices (requested {ndev})")
assert len(devices) == ndev

# Now run test modules 
exit_code = pytest.main(["-s", "block_cyclic_test.py", "-v"])
sys.exit(exit_code)
