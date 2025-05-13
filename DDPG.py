from pathlib import Path
import numpy as np
from numpy.core.multiarray import array as array
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import adamw
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass
from copy import deepcopy

from ..Abstract import BaseAgent, VanillaBatch

from .networks import MLPAC

@dataclass
class DDPGParams:
    obs_dim:int
    act_dim:int
    pi_hidden_sizes:list[int]
    q_hidden_sizes:list[int]
    activation_function:type[nn.Module]
    act_limit:float
    optim_pi:type[optim.Optimizer]
    optim_q:type[optim.Optimizer]
    lr_pi:float
    lr_q:float
    device:str
    gamma:float
    tau:float # Tau for polyak averaging of target network parameters
    noise_scale:float # scaling factor for noise added to actions
    n_steps:int


class DDPG(BaseAgent):

    def __init__(self, hparams: DDPGParams):
        super().__init__(hparams)
        self.device = torch.device(hparams.device)
        self.act_dim = hparams.act_dim
        self.act_limit = hparams.act_limit
        self.tau = hparams.tau
        self.noise_scale = hparams.noise_scale
        self.ac = MLPAC(
            hparams.obs_dim, hparams.act_dim,
            hparams.pi_hidden_sizes, hparams.q_hidden_sizes,
            hparams.act_limit, hparams.activation_function
            ).to(self.device)
        self.target_ac = deepcopy(self.ac).to(self.device)
        self.gamma = hparams.gamma
        self.q_optimizer = hparams.optim_q(self.ac.q.parameters(), lr=hparams.lr_q)
        self.pi_optimizer = hparams.optim_pi(self.ac.pi.parameters(), lr=hparams.lr_pi)
        self.n_steps = hparams.n_steps

    def act(self, obs: np.ndarray, *args, **kwargs):
        action = self.ac.act(torch.as_tensor(obs, device=self.device))
        action += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, batch: VanillaBatch) -> tuple[float, float]:
        self.q_optimizer.zero_grad()
        q_loss = self._compute_loss_q(batch)
        q_loss.backward()
        self.q_optimizer.step()

        # Freeze Critic parameters to speed up eval
        for p in self.ac.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        pi_loss = self._compute_loss_pi(batch)
        pi_loss.backward()
        self.pi_optimizer.step()

        # Unfreeze critic params
        for p in self.ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_target in zip(self.ac.parameters(), self.target_ac.parameters()):
                p_target.data.mul_(1-self.tau)
                p_target.data.add_(self.tau * p.data)
        return pi_loss.cpu().item(), q_loss.cpu().item()

    def _compute_loss_q(self, batch:VanillaBatch) -> tuple[torch.Tensor, np.ndarray]:
        obs = torch.Tensor(batch.observations).to(self.device)
        actions = torch.Tensor(batch.actions).to(self.device)
        rewards = torch.Tensor(batch.rewards).to(self.device)
        next_observations = torch.Tensor(batch.next_observations).to(self.device)
        dones = torch.Tensor(batch.dones).to(self.device)

        with torch.no_grad():
            q_pi_target = self.target_ac.q(next_observations, self.target_ac.pi(next_observations))
            backup = rewards + (self.gamma**self.n_steps) * (1 - dones) * q_pi_target
        
        q = self.ac.q(obs, actions)

        elwise_loss = (q - backup)**2
        q_loss = elwise_loss.mean()
        
        return q_loss
    
    def _compute_loss_pi(self, batch: VanillaBatch) -> torch.Tensor:
        obs = torch.Tensor(batch.observations).to(self.device)
        q_pi = self.ac.q(obs, self.ac.pi(obs))
        return -q_pi.mean()
    
    def save_checkpoint(self, dir: Path, suffix: str = ""):
        filepath = Path(f"ddpg_checkpoint{suffix}.pt")  # 保存在根目录
        torch.save({
            "actor": self.ac.pi.state_dict(),
            "critic": self.ac.q.state_dict(),
            "target_actor": self.target_ac.pi.state_dict(),
            "target_critic": self.target_ac.q.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict()
        }, filepath)
        print(f"Model saved: {filepath}")

    def load_checkpoint(self, dir: Path, suffix: str = ""):
        filepath = Path(f"ddpg_checkpoint{suffix}.pt")  # 从根目录加载
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac.pi.load_state_dict(checkpoint["actor"])
        self.ac.q.load_state_dict(checkpoint["critic"])
        self.target_ac.pi.load_state_dict(checkpoint["target_actor"])
        self.target_ac.q.load_state_dict(checkpoint["target_critic"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        print(f"Model loaded: {filepath} ")

    def load_state_dict(self, state_dict: dict):
        pass

    def get_params(self) -> dict:
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def _compute_loss(self, batch: VanillaBatch) -> tuple[torch.Tensor]:
        pass
        