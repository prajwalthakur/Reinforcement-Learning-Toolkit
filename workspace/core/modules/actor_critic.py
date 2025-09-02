from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

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
        
        
        # Action noise
        self.noise_std_type  = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std*torch.ones(num_actions))
        elif self.noise_std_type == "log":
            # can choose unconstrained number log_std
            # exp(self.log_std) will be always positive
            self.log_std = nn.Parameter(torch.log(init_noise_std*torch.ones(num_actions)))
        else:
            raise ValueError(f"Unkown standard deviation type: {self.noise_std_type}, should be scalr or log")
        
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False) # disable args validation for speedup
        
    def reset(self,dones=None):
        pass     
        
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    #TODO:
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self,obs):
        #compute mean
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unkown standard deviation type:{self.noise_Std_type}, should be scalar or log")
        # create distrbution
        self.distribution = Normal(mean,std)
        
    # during training
    def act(self,obs,**kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalization(obs)
        self.update_distribution(obs)
        return self.distribution.sample(obs)
    
    
    # during evaluation or infrence
    def act_infrence(self,obs,**kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalization(obs)
        return self.actor(obs)
     
     
    # call critic 
    def evaluate(self,obs,**kwargs):
        obs = self.get_critic_obs(obs)
        obs  = self.critic_obs_normalization(obs)
        return self.critic(obs)
    
    
    # filter out the actor observation
    
    def get_actor_obs(self,obs,**kwargs):
        obs_list = []
        for obs_group in self._obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) # batch_size * num_action_obs
    
    
    # filter out the critic observation
    def get_critic_obs(self,obs,**kwargs):
        obs_list = []
        for obs_group in self._obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) # batch_size * num_action_obs
    
    #TODO:
    def get_action_log_prob(self, actions):
        return self.distribution()
    
    
    def update_normalization(self,obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
    
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
           
        