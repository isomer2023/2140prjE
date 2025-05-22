import itertools
from pathlib import Path
from typing import Any

from numpy.core.multiarray import array as array
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from AgentEmporium.Abstract import PrioritizedBatch

from ..Abstract import BaseAgent, VanillaBatch

from .networks import SACActorCritic


@dataclass
class SACHParams:
    gamma:float
    tau:float
    lr_pi:float
    lr_q:float
    alpha:float
    obs_dim:int
    act_dim:int
    act_limit:float
    hidden_pi:list[int]
    hidden_q:list[int]
    activation:type[nn.Module]
    log_std_min:float
    log_std_max:float
    optim_pi:type[optim.Optimizer]
    optim_q:type[optim.Optimizer]
    n_steps:int # for multistep learning
    auto_entropy:bool
    device:str

class SAC(BaseAgent):

    def __init__(self, params:SACHParams):
        super().__init__(params)
        self.gamma = params.gamma
        self.tau = params.tau
        self.lr_pi = params.lr_pi
        self.lr_q = params.lr_q
        self.obs_dim = params.obs_dim
        self.act_dim = params.act_dim
        self.act_limit = params.act_limit
        self.log_std_min = params.log_std_min
        self.log_std_max = params.log_std_max
        self.device = torch.device(params.device)
        self.n_steps = params.n_steps

        self.auto_entropy = params.auto_entropy
        if self.auto_entropy:
            self.target_entropy = -self.act_dim
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )
            self.alpha = self.log_alpha.exp().cpu().item()
            self.alpha_optim = optim.AdamW([self.log_alpha], lr=self.lr_q)
        else:
            self.alpha = params.alpha

        self.ac = SACActorCritic(
            self.obs_dim, self.act_dim, self.act_limit,
            params.hidden_pi, params.hidden_q, params.activation, 
            self.log_std_min, self.log_std_max
            ).to(self.device)
        
        self.target_ac = deepcopy(self.ac).to(self.device)

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Freeze target ac parameters
        for p in self.target_ac.parameters():
            p.requires_grad = False
        
        self.pi_optim = params.optim_pi(self.ac.pi.parameters(), self.lr_pi)
        self.q_optim = params.optim_q(self.q_params, self.lr_q)


    def _compute_loss_q(self, batch:VanillaBatch) -> torch.Tensor:
        tensorize = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.device)
        obs = tensorize(batch.observations, torch.float32)
        act = tensorize(batch.actions, dtype=torch.float32)
        reward = tensorize(batch.rewards, dtype=torch.float32)
        next_obs = tensorize(batch.next_observations, dtype=torch.float32)
        done = tensorize(batch.dones, torch.int8)

        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        # Bellman backup
        with torch.no_grad():
            next_action, logp_next_action = self.ac.pi(next_obs)

            # Target qs
            target_q1 = self.target_ac.q1(next_obs, next_action)
            target_q2 = self.target_ac.q2(next_obs, next_action)
            q_target = torch.min(target_q1, target_q2)

            target = reward + (self.gamma**self.n_steps) * (1 - done) * (q_target - self.alpha * logp_next_action)

        # MSE loss with bellman target
        elwise_loss = ((q1 - target)**2) + ((q2 - target)**2)

        q_loss = torch.mean(elwise_loss)

        return q_loss
    
    def _compute_loss_pi(self, batch:VanillaBatch) -> tuple[torch.Tensor, torch.Tensor]:
        tensorize = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.device)
        obs = tensorize(batch.observations, dtype=torch.float32)

        pi, logp_pi = self.ac.pi(obs)
        q1_pi = self.ac.q1(obs, pi)
        q2_pi = self.ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy regularised policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        if self.auto_entropy:
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            loss_alpha = torch.Tensor(0)

        return loss_pi, loss_alpha
    
    def update(self, batch: VanillaBatch) -> tuple[Any, Any]:
        # Update q1 and q2
        self.q_optim.zero_grad()
        q_loss = self._compute_loss_q(batch)
        q_loss.backward()
        self.q_optim.step()

        # Freeze q parameters
        for p in self.q_params:
            p.requires_grad = False
        
        # Run pi update
        self.pi_optim.zero_grad()
        pi_loss, alpha_loss = self._compute_loss_pi(batch)
        pi_loss.backward()
        self.pi_optim.step()

        if self.auto_entropy:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.cpu().exp().item()

        # Unfreeze q params
        for p in self.q_params:
            p.requires_grad = True

        # Polyak averaging
        with torch.no_grad():
            for p, target_p in zip(self.ac.parameters(), self.target_ac.parameters()):
                target_p.mul_(1 - self.tau)
                target_p.add_(self.tau * p)
        return pi_loss.cpu().item(), q_loss.cpu().item()
    
    def act(self, obs: np.ndarray, *args, deterministic:bool=False, **kwargs):
        return self.ac.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device), deterministic)
    
    def _compute_loss(self, batch: PrioritizedBatch) -> tuple[torch.Tensor]:
        pass

    def save_checkpoint(self, dir: Path, suffix: str = ""):
        pass

    def load_checkpoint(self, dir: Path, suffix: str = ""):
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def get_params(self) -> dict:
        pass

    def eval(self):
        pass

    def train(self):
        pass


        
        