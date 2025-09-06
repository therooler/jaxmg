# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An end-to-end example demonstrating the use of the JAX FFI with CUDA.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os
import ctypes

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi
import os
from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding
from src.jaxmg.potrf import potrf
from jaxmg.cyclic_1d import calculate_padding, calculate_valid_T_A

devices = jax.devices("gpu")



def main():
    ndev = 2
    N = 8
    T_A = 3
    shard_size = N // ndev
    padding = calculate_padding(shard_size, T_A, ndev)
    shard_size_padded = shard_size + padding
    num_tiles = (N + T_A - 1) // T_A
    print(num_tiles)
    dev = 1
    global_blk_id = dev
    tiles = []
    local_idx = 0
    while global_blk_id < num_tiles:
        g_start = global_blk_id * T_A
        print("gs", g_start)
        print("global_blk_id" , global_blk_id)
        g_end = min(g_start + T_A, N)
        T_A_clip = g_end - g_start
        print("T_A_clip", T_A_clip)
        # T_A_clip = min((global_blk_id +1) * T_A, N_A) - global_blk_id * T_A;
        l_start = local_idx * T_A
        l_end = l_start + T_A_clip
        tiles.append((l_start, l_end, g_start, g_end))
        local_idx += 1
        global_blk_id += ndev
    print(tiles)
        # dev = mod_dev % ndev
        # tile_end = min(tile_start + T_A, N)
        # print(f"dev {dev}: {i[dev] * T_A } - {i[dev] * T_A + (tile_end - tile_start)}, idx {tile_start}-{tile_end}")
        # mod_dev += 1
        # i[dev] += 1



if __name__ == "__main__":
    main()
   
