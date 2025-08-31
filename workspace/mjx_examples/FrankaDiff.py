#@title Import MuJoCo, MJX, and Brax

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8" # 0.9 causes too much lag. 
from datetime import datetime
import functools

# Math
import jax.numpy as jnp
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
import brax_envs  # this imports from your local module
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
# Load MJCF XML



# if 'renderer' not in dir():
#     renderer = mujoco.Renderer(mj_model)
# mj_data = mujoco.MjData(mj_model)
# viewer.launch(mj_model, mj_data)



# init_q = mj_model.keyframe('standing').qpos
# mj_data.qpos = init_q
# mujoco.mj_forward(mj_model, mj_data)
# renderer.update_scene(mj_data)



# media.show_image(renderer.render())


# pdb.set_trace()


# Convert to MJX (JAX-friendly)
# mjx_model = mjx.put_model(mj_model)
# mjx_data = mjx.make_data(mjx_model)




def rollout_us(step_env,state,us):
    def step(state,u):
        state  = step_env(state,u)
        return state, (state.reward, state.pipeline_state)
    
    _,(rews,pipeline_states) = jax.lax.scan( step, state, us) 
    return rews, pipeline_states


class CemModel:
    def __init__(self,env,args=None):
        
        self.args = args
        self.env  = env 
        self.nu = env.action_size
        self.num_rollout = 40
        # node to u
        # self.ctrl_dt = 0.02
        # self.step_us = jnp.linspace(0,self.ctrl_dt*)
        self.rollout_us = jax.jit(functools.partial(rollout_us,self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None,0))) #initial-state x u
    
        self.reset_env = jax.jit(env.reset)
        self.step_env = jax.jit(env.step)
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def step_us(self,state:State,action:jax.Array)->jax.Array:
        rews,pipline_states = self.rollout_us_vmap(state,action)
        return rews,pipline_states
     
    @functools.partial(jax.jit, static_argnums=(0,))   
    def cost_function(self,):
        pass


# Get the environment
env = brax_envs.get_environment('franka')

cemmodel = CemModel(env=env)
# Reset with a PRNG key # not used yet
key, subkey = jax.random.split(jax.random.PRNGKey(0))
state_init = cemmodel.reset_env(key)
# Run 100 simulation steps with zero actions


# pdb.set_trace()
# action = jnp.zeros(env.action_size)  # shape = [nu]
# ss
for i in range(100):
    print(i)
    random_actions_seq = random_actions_seq = jax.random.uniform(
            subkey,
            shape=(cemmodel.num_rollout, 5, env.action_size),  # 10 actions, each of dimension action_size
            minval=env.action_range_min,
            maxval=env.action_range_max
        )
    _,state = cemmodel.step_us(state_init, random_actions_seq)
    #print("Reward:", state.reward)