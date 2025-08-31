from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal

from core.networks import MLP
from core.networks import EmpiricalNormalization


class ActorCritic(nn.Module):
    """
    Actor-Critic policy
    Actor: parameterized policy that chooses action
    Critic: Value function that estimates how good a state ( state-action ) is
    
    """
    isRecurrent = False 
    def __init__(    
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims = [256,256,256],
        critic_hidden_dims = [256,256,256],
        activation = "elu",
        init_noise_std = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
        ):
        if kwargs:
            print("ActorCritic.__init_got unexpeceted argument, going to be ignored"
                  +str([key for key in kwargs.keys()]))
        super.__init__() #initialize nn.Module
        
        
        # Actor and Critic inputs are often different. The Critic may use additional
        # information to provide more accurate value estimates, which stabilizes and
        # accelerates training. The Actor, however, is restricted to observations
        # available at deployment to ensure the learned policy is realistic.
        self._obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic Module only support 1d Observations"  #batch-size, feature dimension
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic Module only support 1d Observations"  #batch-size, feature dimension
            num_critic_obs+= obs[obs_group].shape[-1]
            
        #actor network definition
        self.actor = MLP(num_actor_obs,num_actions,actor_hidden_dims,activation)
        
        # actor observation normalization
        #item-6 : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer= EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer= torch.nn.Identity()
        print(f"Actor-MLP: {self.actor}")
        
        
        #Critic
        self.critic = MLP(num_critic_obs,1,critic_hidden_dims,activation)
        self.critic_obs_normalization  = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic-MLP: {self.critic}")
        
            
        
        
            
        