import numpy as np
from collections import deque

from ..Abstract import BaseReplayBuffer, VanillaBatch

class ReplayBuffer(BaseReplayBuffer):

    def __init__(self, max_size:int, obs_dim:int, gamma:float, *args,
                 N_steps:int=1, **kwargs) -> None:
        """
        Initialize the Replay Buffer.

        Args:
            max_size (int): Size of replay buffer, large value recommended (e.g. 1e6)
            obs_dim (int): Size of observations (assumed to be 1D).
            gamma (float): Discount factor
            N_steps (int, optional): For N step learning.
                Defaults to 1 (disabled).
        """
        self.storage = []
        
        self.obs_dim = obs_dim
        self.max_size = max_size

        # N Step Deque
        self.gamma = gamma
        self.N_steps = N_steps
        self.with_Nstep = True if self.N_steps > 1 else False
        if self.N_steps > 1:
            self.n_step_buffer = deque(maxlen=self.N_steps)

        # Trackers
        self.ptr = 0
        self.size = 0
        self.n_eps = 0


    def add(self, obs:np.ndarray, action:np.ndarray, reward:float, next_obs:np.ndarray, done:bool):
        """
        Store a transition in the replay buffer. 

        Args:
            obs (np.ndarray): Observation before the agent takes its action.
            action (np.ndarray): Action taken by the agent.
            reward (float): Reward provided by the environment.
            next_obs (np.ndarray): Observation after the agent has made its action.
            done (bool): Whether the action ended the episode (truncated or terminated).

        Returns:
            tuple: N step transition (if N_step > 1)
        """
        transition = (obs, action, reward, next_obs, done)
        
        if self.with_Nstep:
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.N_steps:
                return ()
            reward, next_obs, done = self.get_n_step_info()
            obs, action = self.n_step_buffer[0][:2]
            # Overwrite existing transition
            transition = (obs, action, reward, next_obs, done)
        if self.ptr >= len(self.storage):
            self.storage.append(transition)
            self.size += 1
        else:
            self.storage[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size
        if done:
            self.n_eps += 1
        
        if self.with_Nstep:
            return self.n_step_buffer[0]

    def sample_from_idcs(self, idcs) -> VanillaBatch:
        """
        Retrieve specific transitions based on their indices. Useful if the random
        sampling must be avoided.

        Args:
            idcs (np.ndarray): Indices where transitions are located.

        Returns:
            dict[str:np.ndarray]: Contains transitions corresponding to idcs.
        """
        observations = np.array([self.storage[idx][0] for idx in idcs], dtype=np.float32)
        actions = np.array([self.storage[idx][1] for idx in idcs], dtype=np.int64)
        rewards = np.array([self.storage[idx][2] for idx in idcs], dtype=np.float32)
        next_observations = np.array([self.storage[idx][3] for idx in idcs], dtype=np.float32)
        dones = np.array([self.storage[idx][4] for idx in idcs], dtype=np.int8)
        return VanillaBatch(observations,
                    actions,
                    rewards,
                    next_observations,
                    dones)

    def sample(self, batch_size:int) -> VanillaBatch:
        """
        Get a batch of stored transitions from the replay buffer. Can have importance
        sampling applied (if alpha > 0), or simply randomly select entries in the
        buffer.

        Returns:
            dict[str:np.ndarray]: Contains a batch of transitions.
        """
        idcs = np.random.randint(low=0, high=len(self.storage) - 1, size=batch_size)
        return self.sample_from_idcs(idcs)
        
    def get_n_step_info(self) -> tuple[float, np.ndarray, bool]:
        """
        Walk through the last N transitions in the N_step buffer (a deque).
        Returns the latest observation, unless the episode ends. Discounts the reward
        from each of the N steps in the buffer.

        Returns:
            float: Discounted reward over N steps
            np.ndarray: Latest non-terminal observation after action was taken.
            bool: Whether the episode was truncated or terminated.
        """
        reward, observation2, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, o2, d = transition[-3:]
            reward = r + self.gamma * reward * (1-d)
            observation2, done = (o2, d) if d else (observation2, done)
        return reward, observation2, done

    def __len__(self):
        return self.size

