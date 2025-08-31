from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain   #chain to combine series of iterables into one iterable


# custom module 

from core.modules import actor_critic
from modules.rnd import RandomNetworkDistillation
from utils import StringToCallable


class PPO:
    policy:actor_critic
    """
        Proximal Policy Optimization
        Policy: Actor-Critic
    """
    def __init__(self):
        pass
    
