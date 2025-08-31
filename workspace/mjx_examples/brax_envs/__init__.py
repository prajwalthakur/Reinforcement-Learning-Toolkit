from .franka_env import FrankaEnv

env_registry = {
    'franka': FrankaEnv,
}

def get_environment(name, config=None):
    return env_registry[name]()
