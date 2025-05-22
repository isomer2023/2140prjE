from __future__ import annotations
import grid2op.Reward
import numpy as np
import torch
import grid2op
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from grid2op.Chronics import MultifolderWithCache
from grid2op.Environment import BaseEnv

VanillaBatch = namedtuple("VanillaBatch", field_names=["observations", "actions",
                          "rewards", "next_observations", "dones"])

class BaseEnvWrapper(ABC):

    def __init__(self, env_name, backend, grid2op_params):
        self.env:BaseEnv = grid2op.make(env_name, backend=backend, **grid2op_params)
        
        if "chronics_class" in grid2op_params.keys():
            if grid2op_params["chronics_class"] is MultifolderWithCache:
                self.env.chronics_handler.set_filter(lambda _: True)
                self.env.chronics_handler.reset()

    @abstractmethod
    def step(self, action:grid2op.Action.baseAction) -> tuple[np.ndarray, float, bool, dict]:
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def seed(self, seed:int) -> None:
        pass

class BaseAgent(ABC):

    def __init__(self, hparams):
        self.hparams = hparams

    @abstractmethod
    def act(self, obs:np.ndarray, *args, **kwargs):
        """Generate an action based on an environment observation

        Args:
            obs (np.array): an observation vector
        """        
        pass

    @abstractmethod
    def save_checkpoint(self, dir:Path, suffix:str=""):
        """Save the weights and biases to file

        Args:
            dir (Path): name of write directory
            suffix (str, optional): suffix for the file name, e.g. "best". Defaults to "".
        """        
        pass

    @abstractmethod
    def load_checkpoint(self, dir:Path, suffix:str=""):
        """Load neural network parameters from checkpoint directory

        Args:
            dir (Path): name of read directory
            suffix (str, optional): e.g. "best". Defaults to "".
        """        
        pass

    @abstractmethod
    def update(self, batch) -> (np.ndarray):
        """
        Take a learning step

        Args:
            batch (dataclass): batch for batch learning

        Returns:
            losses / priorities [np.array]: loss / new weights for each sample
        """        
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict:dict):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return the neural network parameters as a dict of numpy arrays

        Returns:
            dict: a state dict in numpy format
        """        
        pass

    @abstractmethod
    def eval(self):
        """
        Puts agent in evaluation mode where it is only used in feedforward.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Puts agent in training mode, where its weights and biases are updated.
        """
        pass

    def __iter__(self):
        """
        Extract all hyperparameters of the DQN.

        Yields:
            dict[str:object]: Primitive Hyperparmeters of the DQN
        """
        for name, value in self.hparams._asdict().items():
            yield name, value

    @abstractmethod
    def _compute_loss(self, batch) -> tuple[torch.Tensor]:
        pass

class BaseReplayBuffer(ABC):

    def __init__(self, max_size, obs_dim, *args, **kwargs):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.size = 0
        self.ptr = 0
    
    @abstractmethod
    def add(self, obs:np.ndarray, action:np.ndarray, reward:float, next_obs:np.ndarray, done:bool) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def sample(self, batch_size:int):
        pass
    
    def __iter__(self):
        yield from getattr(self, "hparams", {}).items()

    def fill_level(self):
        return self.size / self.max_size