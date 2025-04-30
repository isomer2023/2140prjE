import json, time

import grid2op
import numpy as np
from typing import Tuple, Type
from pathlib import Path

import torch
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Episode import CompactEpisodeData
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import RESET_OPTIONS_TYPING
from grid2op.Reward import BaseReward, GameplayReward

from L2RPN.Abstract import BaseEnvWrapper
from pyexpat import features

from .Utilities import VanillaTracker

class TemplateEnvWrapper(BaseEnvWrapper):

    def __init__(
        self,
        env_name:str,
        backend:Backend,
        env_kwargs:dict,
        *args,
        rho_threshold:float=0.95,
        verbose:bool=False,
        **kwargs
    ):
        if not isinstance(backend, Backend):
            try:
                backend = backend()
            except TypeError:
                print("A backend instance was not provided and initialisation failed. Please provide a valid backend or backend class")
                return
        super().__init__(env_name=env_name, backend=backend, grid2op_params=env_kwargs)
        # >> Simple Attributes <<
        # Used for tracking variables across an episode / step
        self.rho_threshold = rho_threshold # Fraction of thermal limit beyond which agent is activated
        self.tracker:VanillaTracker = VanillaTracker() # Utility to help keep track of reward, observation, etc.
        self.verbose = verbose

    def get_env_size(self) -> tuple[int, list[str]]:
        """
        Deduces the number of episodes present in a locally-stored Grid2Op Environment.

        Args:
            env (grid2op.Environment): Path on disk where Environment is stored

        Returns:
            int: Number of unique episodes in the Environment's chronics
            list[int] (optional): List of Episode IDs
        """
        ids = [x.stem for x in Path(self.env.chronics_handler.path).glob("*") if x.is_dir()]
        return len(ids), ids

    def process_agent_action(self, action) -> BaseAction:
        """
        2025-04-30
        修正充放电方向、效率补偿，并防止过充/过放越界
        - act > 0：电网 -> 电池（充电，power_env > 0）
        - act < 0：电池 -> 电网（放电，power_env < 0）
        """
        obs = self.tracker.state
        # 假设环境步长为 1 小时；若非，则从 env 或配置中读 delta_t
        delta_t = 1.0

        battery_params = [
            {
                "max_charge_power": 7.5,  # MW
                "max_discharge_power": 7.5,  # MW
                "capacity": 15.0,  # MWh
                "charge_eff": 0.95,
                "discharge_eff": 0.95
            },
            {
                "max_charge_power": 3.5,
                "max_discharge_power": 3.5,
                "capacity": 7.0,
                "charge_eff": 0.95,
                "discharge_eff": 0.95
            }
        ]

        action = np.clip(action, -1.0, 1.0)
        set_storage = []

        for i, act in enumerate(action):
            params = battery_params[i]
            cap = params["capacity"]
            soc = obs.storage_charge[i] / cap  # 0 … 1

            # 1) 根据 act 计算网侧或电池侧理想功率（不做限制）
            if act > 0:
                # 充电：网侧给出的功率
                p_grid = act * params["max_charge_power"]
                # 电池实际吸收
                ideal_power = p_grid * params["charge_eff"]
            elif act < 0:
                # 放电：电池端释放功率
                p_batt = (-act) * params["max_discharge_power"]
                # 实际网侧接收
                ideal_power = - p_batt * params["discharge_eff"]
            else:
                ideal_power = 0.0

            # 2) 能量增量 (MWh)
            delta_e = ideal_power * delta_t

            # 3) 过放/过充限制在 [−当前能量, 容量−当前能量]
            e_now = obs.storage_charge[i]
            e_min = - e_now
            e_max = cap - e_now
            safe_delta_e = float(np.clip(delta_e, e_min, e_max))

            # 4) 实际功率，再次校验不超过单步最大功率
            actual_power = safe_delta_e / delta_t
            if actual_power > 0:
                actual_power = min(actual_power, params["max_charge_power"])
            else:
                actual_power = max(actual_power, -params["max_discharge_power"])

            # 如果你的环境对符号有特殊约定，这里可以翻转：
            power_env = actual_power
            # power_env = -actual_power  # 如果环境“正值为放电、负值为充电”，打开这一行

            set_storage.append([i, power_env])

            # Debug 打印，方便查崩溃前的 SOC／功率值
            print(f"[Step Debug] B{i}: SOC={e_now:.3f} MWh, ideal_P={ideal_power:.3f} MW, "
                  f"safe_dE={safe_delta_e:.3f} MWh -> actual_P={power_env:.3f} MW")

        return self.env.action_space({"set_storage": set_storage})

    def convert_observation(self, observation:BaseObservation) -> np.ndarray:
        features = np.concatenate([
            observation.storage_charge,
            observation.rho,
            observation.gen_p,
            observation.gen_q,
            observation.load_p,
            observation.load_q,
            observation.gen_v
        ])
        """
                Convert a Grid2Op observation of the environment's state into a form the
                agent can understand.

                Args:
                    observation (BaseObservation): Observation of grid's state

                Returns:
                    numpy.ndarray: Numpy array that is the input to the agent (will be converted to torch inside the agent)
                """
        return features.astype(np.float32)


    
    def step(self, agent_action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Provides an interface to interact with the wrapped Grid2Op environment.

        Args:
            action (int): integer representation of the action

        Returns:
            tuple[np.ndarray, float, bool, dict]: 
            Observation: vector or graph representation of the environment state
            reward: total accumulated reward over the non active timesteps
            done: if the scenario finished
            info: regular grid2op info. additionally provides a mask for illegal actions, as well as the number of steps taken in the grid2op environment
        """        
        self.tracker.reset_step()

        action = self.process_agent_action(agent_action)
        
        # >> Execute Agent's Action <<
        obs, reward, done, info = self.env.step(action)
        self.tracker.step(obs, reward, done, info)
        
        # >> Step While Safe <<
        # Step through environment so long as line loading is under threshold
        self._step_while_safe()

        self.tracker.info.update({"time":time.perf_counter() - self.tracker.start})
        obs_vec = self.convert_observation(self.tracker.state)

        terminated, truncated = self._get_terminated_truncated()
        return (obs_vec, # Vector Representation of the Observation
                self.tracker.tot_reward, # Reward accumulated, can be a sum if we include heuristics, otherwise is just the reward from env.step(...)
                terminated, # Whether the episode was prematurely ended, i.e. if there's a blackout or powerflow diverges
                truncated, # Whether the episode was truncated, i.e. the agent reached the max time steps for the environment
                self.tracker.info # Additional information, stored in a dictionary
        ) 
        
    def _step_while_safe(self):
        """
        Keep stepping through environment until agent is activated again (or episode ends)
        """
        while not self.tracker.done and not np.any(self.tracker.state.rho >= self.rho_threshold):
            action_ = self.env.action_space({}) # Do Nothing
            obs, reward, done, info = self.env.step(action_)
            self.tracker.step(obs, reward, done, info)


    def reset(self, seed:int|None=None, options:RESET_OPTIONS_TYPING={}) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        """
        Reset the environment, this will start a new episode
        
        Returns:
            np.ndarray | Data | Any: observation, type depends
                on conversion routine.
            dict[str]: Information, contains the following:
                "action_mask": None
                "steps_taken": int
                "reward": 0.0
            bool: Whether the episode has been terminated/truncated,
                should be False.
        """
        # >> Episode ID <<
        if "time serie id" in options:
            ep_id = options["time serie id"]
        else:
            ep_id = self.env.chronics_handler.get_name()
            options["time serie id"] = ep_id
        
        # >> Reset Environment to Target Episode <<
        # NOTE: Options can overwrite the init_ts
        self.tracker.reset_episode(
            self.env.reset(options=options)
        )
        
        obs_vec = self.convert_observation(self.tracker.state)
        terminated, truncated = self._get_terminated_truncated()
        return (
            obs_vec, # Obs
            dict(reward=0), # Info
            terminated, truncated
            )
    
    
    def _get_terminated_truncated(self) -> Tuple[bool, bool]:
        """
        Terminated: Episode ended prematurely (game over)
        Truncated: Episode ended because we reached the end of the timeseries (win!)

        Returns:
            Tuple[bool, bool]: Terminated, Truncated
        """
        done = self.tracker.done
        step = self.env.nb_time_step
        env_max_step = self.env.max_episode_duration()
        terminated = done and not (step == env_max_step)
        truncated = done and (step == env_max_step)
        return terminated, truncated

    def set_id(self, chronic_id:int|str):
        self.env.set_id(chronic_id)
    
    def seed(self, seed:int) -> None:
        self.env.seed(seed=seed)

    def max_episode_duration(self) -> int:
        return self.env.max_episode_duration()
            
        
