import itertools
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .networks import TD3AC

from ..Abstract import BaseAgent, VanillaBatch

@dataclass
class TD3HParams:
    obs_dim:int
    act_dim:int
    act_limit:int
    pi_hidden:list[int]
    q_hidden:list[int]
    activation:type[nn.Module]
    optim_pi:type[optim.Optimizer]
    optim_q:type[optim.Optimizer]
    lr_pi:float
    lr_q:float
    device:str
    gamma:float
    tau:float # polyak averaging strength
    act_noise:float
    target_noise:float
    policy_delay:int
    noise_clip:float
    n_steps:int


class TD3(BaseAgent):

    def __init__(self, hparams: TD3HParams):
        super().__init__(hparams)
        self.device = torch.device(hparams.device)
        self.ac = TD3AC(
            hparams.obs_dim, hparams.act_dim, hparams.act_limit, hparams.activation,
            hparams.pi_hidden, hparams.q_hidden
        ).to(self.device)
        self.act_dim = hparams.act_dim
        self.act_noise = hparams.act_noise
        self.tau = hparams.tau
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.target_ac = deepcopy(self.ac).to(self.device)
        self.target_noise = hparams.target_noise
        self.noise_clip = hparams.noise_clip
        self.policy_delay = hparams.policy_delay
        self.timer = 0
        self.gamma = hparams.gamma
        self.act_limit = hparams.act_limit
        self.pi_optimizer = hparams.optim_pi(self.ac.pi.parameters(), lr=hparams.lr_pi)
        self.q_optimizer = hparams.optim_q(self.q_params, lr=hparams.lr_q)
        self.n_steps = hparams.n_steps
        # freeze target params:
        for p in self.target_ac.parameters():
            p.requires_grad = False

    def act(self, obs: np.ndarray, *args, with_noise:bool=True, **kwargs):
        act = self.ac.act(torch.as_tensor(obs, device=self.device))
        if with_noise:
            act += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(act, -self.act_limit, self.act_limit)

    def _compute_loss_pi(self, batch:VanillaBatch) -> torch.Tensor:
        obs = torch.as_tensor(batch.observations, device=self.device)
        q1_pi = self.ac.q1(obs, self.ac.pi(obs))
        return -q1_pi.mean()

    def _compute_loss_q(self, batch:VanillaBatch) -> torch.Tensor:
        tensorize = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.device)
        obs = tensorize(batch.observations, dtype=torch.float32)
        act = tensorize(batch.actions, dtype=torch.float32)
        reward = tensorize(batch.rewards, dtype=torch.float32)
        next_obs = tensorize(batch.next_observations, dtype=torch.float32)
        done = tensorize(batch.dones, dtype=torch.int8)

        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        # Bellman backup
        with torch.no_grad():
            pi_target = self.target_ac.pi(next_obs)

            # target policy smoothing
            epsilon = torch.randn_like(pi_target) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)

            next_act = pi_target + epsilon
            next_act = torch.clamp(next_act, -self.act_limit, self.act_limit)

            # target q vals
            target_q1 = self.target_ac.q1(next_obs, next_act)
            target_q2 = self.target_ac.q2(next_obs, next_act)

            q_pi_target = torch.min(target_q1, target_q2)

            target = reward + (self.gamma ** self.n_steps) * (1 - done) * q_pi_target
        # q-losses
        elwise_loss = ((q1 - target)**2 + (q2 - target)**2) / 2

        q_loss = elwise_loss.mean()

        return q_loss


    def save_checkpoint(self, dir: Path, suffix: str = ""):
        pass

    def load_checkpoint(self, dir: Path, suffix: str = ""):
        pass

    def update(self, batch:VanillaBatch) -> tuple[None|float, float]:
        # Q-update
        self.q_optimizer.zero_grad()
        q_loss = self._compute_loss_q(batch)
        q_loss.backward()
        self.q_optimizer.step()

        # Update pi and target networks with delay
        pi_loss = None
        if self.timer % self.policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            pi_loss = self._compute_loss_pi(batch)
            pi_loss.backward()
            self.pi_optimizer.step()

            for p in self.q_params:
                p.requires_grad = True
            
            with torch.no_grad():
                for target_p, p in zip(self.target_ac.parameters(), self.ac.parameters()):
                    target_p.mul_(1 - self.tau)
                    target_p.add_(self.tau * p)
        self.timer += 1
        if pi_loss is not None:
            pi_loss = pi_loss.cpu().item()
        return pi_loss, q_loss.cpu().item()



    def load_state_dict(self, state_dict: dict):
        pass

    def get_params(self) -> dict:
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def _compute_loss(self, batch: VanillaBatch):
        pass


