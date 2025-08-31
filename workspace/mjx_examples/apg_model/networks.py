

"""APG networks"""



from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PolicyParams,Policy
from brax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class APGNetworks:
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution

def make_inference_fn(apg_netwrks:APGNetworks):
    """Creates params and inference function for the APG agent."""
    def make_policy(
        params: PolicyParams,determinnistic:bool=False 
    )->Policy :
        def policy( 
                observations: types.Observation, key_sample:PRNGKey
            
        )->Tuple[types.Action,types.Extra]:
            logits = apg_netwrks.policy_network.apply(*params,observations)
            if determinnistic:
                return apg_netwrks.parametric_action_distribution.mode(logits),{}
            else:
                return(
                    apg_netwrks.parametric_action_distribution.sample(
                        logits,key_sample
                    ),{},
                )
        return policy
    return  make_policy

def make_apg_networks(
    
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn= types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (32,)*4,
    activation: networks.ActivationFn = linen.elu,
    layer_norm: bool = True,
) -> APGNetworks:
    "MAKE APG NETWORK"
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size= action_size, var_scale=0.1
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        kernel_init=linen.initializers.orthogonal(0.01),
        layer_norm=layer_norm,
    )   
    return APGNetworks(
        policy_network=policy_network,
        parametric_action_distribution = parametric_action_distribution,
    )