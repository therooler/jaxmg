from jax.sharding import NamedSharding


def get_mesh_and_spec_from_array(A):
    sharding = A.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


def check_matrix_validity(matrix_size, num_devices):
    if matrix_size % num_devices != 0:
        raise ValueError(
            f"Matrix of size N x N must be have N divisible by number of devices {num_devices}, receieved N = {matrix_size}."
        )
