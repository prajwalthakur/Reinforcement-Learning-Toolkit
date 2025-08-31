# ruff: noqa: ANN001, ANN202, FIX002, ERA001, PLR0913, PLR0914, PLR0915, PLR0917, TD003

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


class FrankaFR3(PipelineEnv):
    """Environment for training FR3 and DEX-EE to reach the target position."""

    def __init__(self, control_timestep: float = 1 / 50, **kwargs: dict) -> None:
        mj_model = create_mj_model()
        sys = mjcf.load_model(mj_model)

        kwargs["n_frames"] = 10
        kwargs["backend"] = "mjx"
        super().__init__(sys, **kwargs)
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)

    def reset(self, rng: jax.Array) -> State:
        data = self.pipeline_init(self._init_q, jnp.zeros(self.sys.nv))  # intit q and qd
        obs, reward, done = jnp.zeros(7), jnp.zeros(()), jnp.zeros(())
        metrics, info = {}, {}
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        data0 = state.pipeline_state
        data1 = self.pipeline_step(data0, action)
        return state.replace(pipeline_state=data1)


def main():
    environment = FrankaFR3()

    rng = jax.random.PRNGKey(0)
    key_env, eval_key, rng = jax.random.split(rng, 3)

    num_envs = 1024
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])

    episode_length = 600
    action_repeat = 1
    env = envs.training.wrap(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )  # pytype: disable=wrong-keyword-args

    num_eval_envs = 128

    def make_policy(params: types.Params, deterministic: bool = False):
        del params

        def policy(
            observations: types.Observation,
            key_sample: types.PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            if deterministic:
                actions = jnp.zeros((*observations.shape[0:-1], env.action_size))
            else:
                actions = jax.random.uniform(
                    key_sample,
                    (*observations.shape[0:-1], env.action_size),
                    minval=-1.0,
                    maxval=+1.0,
                )
            return actions, {}

        return policy

    evaluator = acting.Evaluator(
        env,
        make_policy,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    metrics = evaluator.run_evaluation(None, training_metrics={})
    print(metrics)


if __name__ == "__main__":
    main()