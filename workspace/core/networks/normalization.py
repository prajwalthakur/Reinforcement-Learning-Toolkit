
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""
    def __init__(self, shape, eps=1e-2, until = None):
        """
            Initialize EmpericalNormalization Module
            Args:
                shape (int or tuple of int ): shape of the input values except the batch size
                eps (float): Small Value for stability
                until (int or None): if this arg specified, the module learns input values until 
                the sum of batch sizes exceeds it
            Note: normalization parameters are computed over whole batch, not for each environment seperately.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0)) # mean dim (1,shape)
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std",torch.ones(shape).unsqueeze(0))
        self.register_buffer("_count",torch.tensor(0,dtype=torch.long))
        
    @property
    def mean(self):
        return self._mean.squeeze(0).clone() # to return the copy of the tensor and not the view
    
    @property
    def std(self):
        return self._std.squeeze(0).clone()  # to return the copy of the tensor and not the view
    
    @property
    def forward(self,x):
        "normalize mean and variance of values based on emperical values"
        norm_val = (x- self._mean)/(self._std + self.eps)
        return norm_val

    @torch.jit.unused
    def update(Self,x):
        "Learn input values without computing the output values of them"
        # updated-mean = old_mean + (new_data- old_mean)/(old_count + count(new_data))
    
    @torch.jit.unused
    def inverse(self,y):
        "De normalize values based on emperical values"
        unorm_val = (self._std+self.eps)*y + self._mean
        return unorm_val