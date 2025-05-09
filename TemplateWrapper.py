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
            env_name: str,
            backend: Backend,
            env_kwargs: dict,
            *args,
            rho_threshold: float,
            verbose: bool = False,
            **kwargs
    ):
        if not isinstance(backend, Backend):
            try:
                backend = backend()
            except TypeError:
                print(
                    "A backend instance was not provided and initialisation failed. Please provide a valid backend or backend class")
                return
        super().__init__(env_name=env_name, backend=backend, grid2op_params=env_kwargs)
        # >> Simple Attributes <<
        # Used for tracking variables across an episode / step
        self.rho_threshold = rho_threshold  # Fraction of thermal limit beyond which agent is activated
        self.tracker: VanillaTracker = VanillaTracker()  # Utility to help keep track of reward, observation, etc.
        self.verbose = verbose

    def get_env_size(self) -> tuple[int, list[str]]:

        ids = [x.stem for x in Path(self.env.chronics_handler.path).glob("*") if x.is_dir()]

        return len(ids), ids

    '''
    def process_agent_action(self, action) -> BaseAction:
    
        
        max_charge = self.env.action_space.storage_max_p_absorb  
        max_discharge = self.env.action_space.storage_max_p_prod  

        storage_actions = []
        for storage_id in [0, 1]:  
            action_val = action[storage_id]

            action_val = np.clip(action_val, -2.0, 2.0)

            if -1 <= action_val < 0:
                storage_actions.append((storage_id, 0.0))
                #discharge_p = action_val * max_charge[storage_id]
                #storage_actions.append((storage_id, discharge_p))  
            
            elif 0 < action_val <= 1:
                storage_actions.append((storage_id, 0.0))
                #charge_p = action_val * max_discharge[storage_id]
                #storage_actions.append((storage_id, charge_p))
            
            else:  
                
                storage_actions.append((storage_id, 0.0))
                #print(f"Battery {storage_id} do nothing.")
                
        action = self.env.action_space({"set_storage": storage_actions})
        return action

        '''


    def convert_observation(self, observation: BaseObservation) -> np.ndarray:
        SOC  = observation.storage_charge/observation.storage_Emax
        rho  = observation.rho/2
        gen  = observation.gen_p/89
        load = observation.load_p/15
        topo = observation.topo_vect
        
        topo_normalized = np.select(
            [topo == -1, topo == 1, topo == 2],
            [0.0, 0.5, 1.0],
            default=0.0
        ).astype(np.float32)

        features = np.concatenate([
            SOC,
            rho,
            gen,
            load,
            topo_normalized
        ])
        return features.astype(np.float32)

    def process_agent_action(self, action) -> BaseAction:

        max_charge = self.env.action_space.storage_max_p_absorb
        max_discharge = self.env.action_space.storage_max_p_prod

        storage_actions = []
        for storage_id in [0, 1]:
            action_val = action[storage_id]

            action_val = np.clip(action_val, -2.0, 2.0)

            if -1 <= action_val < 0: # Charge

                charge_p = abs(action_val) * max_charge[storage_id]
                storage_actions.append((storage_id, charge_p))
                #storage_actions.append((storage_id, 0.0))

            elif 0 < action_val <= 1: # Discharge

                discharge_p = action_val * max_discharge[storage_id]
                storage_actions.append((storage_id, -discharge_p))
                #storage_actions.append((storage_id, 0.0))

            else:

                storage_actions.append((storage_id, 0.0))
                #print(f"Battery {storage_id} do nothing.")

        action = self.env.action_space({"set_storage": storage_actions})
        return action





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

        self.tracker.info.update({"time": time.perf_counter() - self.tracker.start})
        obs_vec = self.convert_observation(self.tracker.state)

        terminated, truncated = self._get_terminated_truncated()
        return (obs_vec,  # Vector Representation of the Observation
                self.tracker.tot_reward,
                # Reward accumulated, can be a sum if we include heuristics, otherwise is just the reward from env.step(...)
                terminated,
                # Whether the episode was prematurely ended, i.e. if there's a blackout or powerflow diverges
                truncated,
                # Whether the episode was truncated, i.e. the agent reached the max time steps for the environment
                self.tracker.info  # Additional information, stored in a dictionary
                )

    def _step_while_safe(self):
        """
        Keep stepping through environment until agent is activated again (or episode ends)
        """
        while not self.tracker.done and not np.any(self.tracker.state.rho >= self.rho_threshold):

            #action_ = self.env.action_space({"set_storage": [(0, 0.0),(1, 0.0)]})

            action_ = self.env.action_space({})  # Do Nothing
            obs, reward, done, info = self.env.step(action_)
            self.tracker.step(obs, reward, done, info)

    def reset(self, seed: int | None = None, options: RESET_OPTIONS_TYPING = {}) -> Tuple[
        np.ndarray, np.ndarray, bool, bool]:
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
            obs_vec,  # Obs
            dict(reward=0),  # Info
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

    def set_id(self, chronic_id: int | str):
        self.env.set_id(chronic_id)

    def seed(self, seed: int) -> None:
        self.env.seed(seed=seed)

    def max_episode_duration(self) -> int:
        return self.env.max_episode_duration()
            
        