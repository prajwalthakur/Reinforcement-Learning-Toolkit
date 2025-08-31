# MLP network 
from __future__ import annotations
import torch
import torch.nn as nn
from functools import reduce

from utils.utils import resolve_nn_activation

class MLP(nn.Sequential):
    """
    Multi Layer Perceptron
    The MLP network is a sequence of linear layers and activation functions. The
    last layer is a linear layer that outputs the desired dimension unless the
    last activation function is specified.

    It provides additional conveniences:

    - If the hidden dimensions have a value of ``-1``, the dimension is inferred
      from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired
      shape.    
    """
    def __init__(
        self,
        input_dim:int,
        output_dim:int| tuple[int]| list[int],
        hidden_dims: tuple[int]| list[int],
        activation: str = "elu",
        last_activation: str|None = None,
    ):
        """
        Initialize MLP
        Args:
            input_dim : Dimension of the input
            output_dim: Dimension of the output
            hidden_dims: Dimension of the hidden layer (i.e number of layers not including input and output)
            activation: Activation function, default to "elu"
            last_activation: Activation function for last layer, default to None,
                            In that case, last layer is a linear layer
        """
        super.__init__()
        
        # convert str activation to respective Activation torch module
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(activation) if activation is not None else None
        
        # resolve number of activation layer
        hidden_dims_processed = [input_dim if dim ==-1 else dim for dim in hidden_dims]
        num_hidden_dims = hidden_dims_processed.shape[-1]
        #create layers sequentially
        #1. input layers
        layers = []
        layers.append(nn.Linear(input_dim,hidden_dims_processed[0]))
        layers.append(activation_mod)
        
        #2. hidden layers
        for i in range(1,num_hidden_dims,1):
            layers.append(nn.Linear(hidden_dims_processed[i-1],hidden_dims_processed[i]))
            layers.append(activation_mod)
        
        #3 last layer
        if isinstance(output_dim,int):
            layers.append(nn.Linear(hidden_dims_processed[-1],output_dim))
        else:
            total_output_dim  = reduce(lambda x,y:x*y, output_dim) #flatten dimension
            layers.append(nn.Linear(hidden_dims_processed[-1],total_output_dim))
            layers.append(nn.Unflatten(output_dim))
            
        
        if last_activation_mod is not None:
            layers.append(last_activation_mod)
        # register module 
        for idx,layer in enumerate(layers):
            self.add_module(f"{idx}",layer)
    
    def init_weights(self,scales: float | tuple[float]):
        """
            initialize the weights of the MLP
        Args:
            scales: Scale factor for the weights.
        """
        def get_scale(idx)->float:
            """get the scale factor for the weights of the MLP
            Args:
                idx: Index of the layer
            """
            return scales[idx] if isinstance(scales,(tuple,list)) else scales
        
        for idx,module in enumerate(self):
            if isinstance(module,nn.Linear):
                # Initializes features in orthogonal directions. 
                # This helps preserve the scale of activations/gradients across layers, 
                # reducing the risk of vanishing or exploding values.
                nn.init.orthogonal_(module.weight,gain=get_scale(idx)) #colons are orthonormal vectors
                nn.init.zeros_(module.bias)
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        """
        forward pass of the MLP.
        
        """
        for layer in self:
            input = layer(input)
        return input