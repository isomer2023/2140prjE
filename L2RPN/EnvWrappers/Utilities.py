import time
from grid2op.Observation import BaseObservation

class VanillaTracker():
    
    def __init__(self):
        # Per-Episode Attributes
        self.t:int = 0
        self.previous_obs:BaseObservation = None
        self.state:BaseObservation = None
        self.reward:float = None
        self.info:dict = {}
        self.done:bool = False
        
        # Per-N-Step Attributes
        self.start:float = time.perf_counter()
        self.tot_reward:float = 0.0
        self.n_steps:int = 0
    
    def reset_episode(self, obs:BaseObservation):
        self.t = 0
        self.previous_obs = None
        self.state = obs
        self.reward = None
        self.info = {}
        self.done = False
        
    def reset_step(self):
        self.start = time.perf_counter()
        self.tot_reward = 0.0
        self.n_steps = 0
        
    def step(self, obs:BaseObservation, reward:float, done:bool, info:dict):
        self.n_steps += 1
        self.t += 1
        self.reward = reward
        self.tot_reward += self.reward
        self.previous_obs = self.state
        self.state = obs
        self.done = done
        self.info = info
