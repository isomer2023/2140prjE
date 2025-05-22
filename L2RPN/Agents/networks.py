import torch, torch.nn as nn, torch.distributions.normal as normal, torch.nn.functional as F
import numpy as np

def mlp(sizes, activation, output_activation:type[nn.Module]=nn.Identity):
    layers = []
    for i in range(len(sizes) -1):
        activation = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(activation())
    return nn.Sequential(*layers)

class MLPActor(nn.Module):

    def __init__(self, obs_dim:int, act_dim:int, hidden_sizes:list[int], activation:type[nn.Module], act_limit:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pi_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pi = mlp(pi_sizes, activation)
        self.act_limit = act_limit

    def forward(self, obs:torch.Tensor):
        return self.act_limit * self.pi(obs)
    
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim:int, act_dim:int, hidden_sizes:list[int], activation:type[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs:torch.Tensor, act:torch.Tensor):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) 
    
class MLPAC(nn.Module):

    def __init__(self, obs_dim:int, act_dim, hidden_sizes_pi:list[int], hidden_sizes_q:list[int], act_limit:float, activation:type[nn.Module]=nn.ReLU, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes_pi, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes_q, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
        

class TD3AC(nn.Module):

    def __init__(self, obs_dim:int, act_dim:int, act_limit:float, activation:type[nn.Module], actor_hidden:list[int], critic_hidden:list[int]):
        super().__init__()
        self.pi = MLPActor(obs_dim, act_dim, actor_hidden, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, critic_hidden, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, critic_hidden, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
        
class SquashedGaussianMlpActor(nn.Module):

    def __init__(self, obs_dim:int, act_dim:int, hidden:list[int], activation:type[nn.Module], act_limit:float, log_std_min:float=-20, log_std_max:float=2):
        super().__init__()
        self.net = mlp([obs_dim] + hidden, activation, activation)
        self.mu_layer = nn.Linear(hidden[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden[-1], act_dim)
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs:torch.Tensor, deterministic:bool=False, with_log_prob:bool=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # pre-squash distribution and sample
        pi_distribution = normal.Normal(mu, std)
        if deterministic: # use at test time
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_log_prob:
            # Compute logprob from gaussian and apply correction for tanh squashing.
            # Correction is _magic_, reasoning is found in App. C of SAC paper. 
            # More numerically stable version here courtesy of OpenAI spinup, equiv. to eq. 21 in paper
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
    

class SACActorCritic(nn.Module):
    
    def __init__(self, obs_dim:int, act_dim:int, act_limit:float, hidden_pi:list[int], hidden_q:list[int], activation:type[nn.Module]=nn.ReLU, log_std_min:float=-20, log_std_max:float=2):
        super().__init__()
        self.pi = SquashedGaussianMlpActor(obs_dim, act_dim, hidden_pi, activation, act_limit, log_std_min, log_std_max)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_q, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_q, activation)

    def act(self, obs:torch.Tensor, deterministic:bool=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
        
    
        
