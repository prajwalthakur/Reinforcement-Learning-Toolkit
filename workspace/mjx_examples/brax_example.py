import jax
import jax.numpy as jnp
import brax_envs  # this imports from your local module

# Get the environment
env = brax_envs.get_environment('franka')

# Reset with a PRNG key
state = env.reset(jax.random.PRNGKey(0))

# Run 100 simulation steps with zero actions
for _ in range(1):
    action = jnp.zeros(env.action_size)  # shape = [nu]
    state = env.step(state, action)
    print("Reward:", state.reward)
