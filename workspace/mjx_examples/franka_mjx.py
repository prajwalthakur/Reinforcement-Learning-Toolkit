#@title Import MuJoCo and MJX (No Brax)

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # avoid OOM
from datetime import datetime
import functools

# Math
import jax.numpy as jnp
import numpy as np
import jax
from jax import config  
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')

# Sim
import mujoco
import mujoco.mjx as mjx
from mujoco import viewer
from pathlib import Path

# Supporting
import matplotlib.pyplot as plt
from typing import Any, Dict

import pdb

# -------------------------------
# Rollout function (JAX-friendly)
# -------------------------------
def rollout_us(mjx_model, state, us):
    def step(state, u):
        # set controls in the mjx.Data
        state = state.replace(ctrl=u)   # JAX-friendly immutable update
        # advance simulation
        next_state = mjx.step(mjx_model, state)
        reward = 0.0  # custom reward
        return next_state, (reward, next_state)

    final_state, (rews, states) = jax.lax.scan(step, state, us)
    return rews, states


# -------------------------------
# Example Model wrapper
# -------------------------------
class MODEL:
    def __init__(self, args, mjx_model):
        self.args = args
        self.mjx_model = mjx_model
        self.nu = mjx_model.nu

        self.rollout_us = jax.jit(functools.partial(rollout_us, self.mjx_model))
        self.rollout_us_vmap = jax.jit(
            jax.vmap(self.rollout_us, in_axes=(None, 0))  # batch rollouts
        )


# -------------------------------
# Main
# -------------------------------
MENAGERIE_ROOT = Path(os.environ["MUJOCOMEN_PATH"])

def main():
    # Load and patch MJCF model
    xml_path = (MENAGERIE_ROOT / "franka_emika_panda" / "mjx_panda.xml").as_posix()
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Optional: disable self-collisions
    mj_model.geom_contype[:] = 0
    mj_model.geom_conaffinity[:] = 0

    # Convert to MJX (JAX-compatible)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    print(mjx_data.qpos, type(mjx_data.qpos), "device= " , mjx_data.qpos.device)

    # Optionally inspect interactively
    # pdb.set_trace()
    viewer.launch(mj_model, mj_data)

    # Rollout with zero actions
    T = 100
    us = jnp.zeros((T, mjx_model.nu))  # control sequence
    rewards, states = rollout_us(mjx_model, mjx_data, us)

    print("Total reward:", jnp.sum(rewards))
    print("Final qpos:", states[-1].qpos)


if __name__ == "__main__":
    main()
