
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp


class ContinualWorldActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        return data

    def get_std(self) -> th.Tensor:
        # Make sure standard deviation is not too small
        # minimal std = exp(LOG_STD_MIN) = exp(-20) approx 2e-9
        return super().get_std()

    def make_actor_net(self) -> nn.Sequential:
        """
        Custom actor network creation to match Continual World reference:
        Layer 1: Linear -> LayerNorm -> Tanh
        Subsequent layers: Linear -> LeakyReLU
        """
        modules = []
        # Input layer
        modules.append(nn.Linear(self.features_dim, self.net_arch[0]))
        modules.append(nn.LayerNorm(self.net_arch[0]))
        modules.append(nn.Tanh())

        # Subsequent layers
        for idx in range(len(self.net_arch) - 1):
             modules.append(nn.Linear(self.net_arch[idx], self.net_arch[idx + 1]))
             modules.append(self.activation_fn())
            
        return nn.Sequential(*modules)


class ContinualWorldCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )

    def _create_critic_net(self, latent_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> nn.Sequential:
        """
        Custom critic network creation to match Continual World reference:
        Layer 1: Linear -> LayerNorm -> Tanh
        Subsequent layers: Linear -> LeakyReLU
        """
        modules = [nn.Linear(latent_dim, net_arch[0]),
                   nn.LayerNorm(net_arch[0]),
                   nn.Tanh()]

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        # Output layer (handled by base class usually, but Base class create_mlp doesn't allow custom first layer easily)
        # Actually ContinuousCritic calls create_mlp. We need to override the q_networks creation.
        
        # SB3 ContinuousCritic implementation uses create_mlp. 
        # To customize per layer, we have to copy the logic or override constructor.
        # But wait, ContinuousCritic calls _create_critic_net for the *body*? 
        # No, it calls create_mlp directly in __init__.
        # So I have to override __init__ or just q_networks.
        
        # Let's see: ContinuousCritic.__init__ does:
        # self.q_networks = []
        # for idx in range(n_critics):
        #    q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        #    self.q_networks.append(nn.Sequential(*q_net))
        
        # I need to reimplement the loop in __init__ basically.
        return nn.Sequential(*modules, nn.Linear(net_arch[-1], 1))

    # To properly override without copy-pasting everything from __init__, we might be stuck
    # because __init__ sets self.q_networks.
    # However, Python allows us to re-set it after super().__init__().
    
    # Wait, simple inheritance approach:
    # call super(), then overwrite self.q_networks
    
    pass 

# Actually, defining the class to perform the override in __init__ is cleaner.

class ContinualWorldCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )
        
        # Re-create q_networks with custom architecture
        action_dim = self.action_space.shape[0] # type: ignore[index]
        self.q_networks = []
        for idx in range(n_critics):
            # Custom matching: 
            # Input -> [Linear, LayerNorm, Tanh] -> [Linear, LeakyReLU] * (depth-1) -> Linear(1)
            
            modules = []
            # First layer
            modules.append(nn.Linear(features_dim + action_dim, net_arch[0]))
            modules.append(nn.LayerNorm(net_arch[0]))
            modules.append(nn.Tanh())
            
            # Subsequent layers
            for i in range(len(net_arch) - 1):
                modules.append(nn.Linear(net_arch[i], net_arch[i + 1]))
                modules.append(activation_fn())

            # Output layer
            modules.append(nn.Linear(net_arch[-1], 1))
            
            q_net = nn.Sequential(*modules)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class ContinualWorldMlpPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        *args,
        **kwargs,
    ):
        # Create a new actor and critic by default
        # We need to force our custom classes
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ContinualWorldActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinualWorldCritic(**critic_kwargs).to(self.device)
