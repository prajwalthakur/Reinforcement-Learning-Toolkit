#@title Import MuJoCo, MJX, and Brax

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8" # 0.9 causes too much lag. 
from datetime import datetime
import functools

# Math
import jax.numpy as jp
import numpy as np
import jax
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')
from brax import math

# Sim
import mujoco
import mujoco.mjx as mjx
from mujoco import viewer
# Brax
from brax import envs
from brax.base import Motion, Transform
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from brax.mjx.pipeline import _reformat_contact
from brax.training.acme import running_statistics
from brax.io import model

# Algorithms
from brax.training.agents.apg import train as apg
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import train as ppo

# Supporting
from etils import epath
import mediapy as media
import matplotlib.pyplot as plt
from ml_collections import config_dict
from typing import Any, Dict

import pdb
# loading the mjx model

xml_path = epath.Path('/root/workspace/src/mujoco_menagerie/anybotics_anymal_c/scene_mjx.xml').as_posix()

mj_model = mujoco.MjModel.from_xml_path(xml_path)

if 'renderer' not in dir():
    renderer = mujoco.Renderer(mj_model)

init_q = mj_model.keyframe('standing').qpos

mj_data = mujoco.MjData(mj_model)
mj_data.qpos = init_q
mujoco.mj_forward(mj_model, mj_data)
renderer.update_scene(mj_data)
pdb.set_trace()
#viewer.launch(mj_model, mj_data)

# media.show_image(renderer.render())


# pdb.set_trace()


# Rendering Rollouts
def render_rollout(reset_fn, step_fn, 
                   inference_fn, env, 
                   n_steps = 200, camera=None,
                   seed=0):
    rng = jax.random.key(seed)
    render_every = 3
    state = reset_fn(rng)
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    media.show_video(env.render(rollout, camera=camera), 
                     fps=1.0 / (env.dt*render_every),
                     codec='gif')