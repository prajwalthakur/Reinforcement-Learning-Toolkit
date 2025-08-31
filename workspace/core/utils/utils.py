from __future__ import annotations

#import git
import importlib
import os
import pathlib
import torch
import warnings
#from tensordict import TensorDict
from typing import Callable

def resolve_nn_activation(act_name:str)-> torch.nn.Module:
    """
    convert activation name to respective torch.nn.Module
    
    Returns:
        torch.nn.Module
    Raises:
        Value Error: if activation function is not defined
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU()
    }
    act_name = act_name.lower() 
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"invalid activation function '{act_name}'. Valid functions are {list(act_dict.keys())}")