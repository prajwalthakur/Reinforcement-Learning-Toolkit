from brax import envs
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import mjcf
from brax.training import acting
from brax.training import types
from etils import epath
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
# from robot_descriptions import fr3_mj_description
import time
from typing import Any, Dict, Sequence, Tuple, Union, List
import pdb

def create_mj_model() -> mujoco.MjModel:
    franka_fr3_path = epath.Path("/root/workspace/src/mujoco_menagerie/franka_emika_panda/scene.xml")
    robot_spec = mujoco.MjSpec.from_file(franka_fr3_path.as_posix())

    for geom in robot_spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0

    robot_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    robot_spec.option.timestep = 1 / 1_000
    robot_spec.option.iterations = 6
    robot_spec.option.ls_iterations = 8
    robot_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    model = robot_spec.compile()

    return model


class FrankaEnv(PipelineEnv):
    def __init__(self, control_timestep: float = 1 / 50, **kwargs: dict) -> None:
        mj_model = create_mj_model()
        sys = mjcf.load_model(mj_model)
        n_frames = int(control_timestep/sys.opt.timestep)
        self._init_q = jnp.array(sys.mj_model.keyframe("home").qpos)
        # number of everything
        self._nv = sys.nv
        self._nq = sys.nq
        self.action_range_min = sys.actuator_ctrlrange[:,0]
        self.action_range_max = sys.actuator_ctrlrange[:,1]
        kwargs["n_frames"] = n_frames
        kwargs["backend"] = "mjx"
        super().__init__(sys, **kwargs)


    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        data = self.pipeline_init(self._init_q, jnp.zeros(self.sys.nv))  # q and qd
        obs, reward, done = jnp.zeros(7), jnp.zeros(()), jnp.zeros(())
        metrics, info = {}, {}
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        state = state.replace(pipeline_state=pipeline_state)
        return state

    # def _get_obs(self,pipeline_sate:State,state_info: dict[str,Any])->jax.Array:
        