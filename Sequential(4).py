"""
Main script for running RL Agent Training.
Complete the pseudocode here as part of your project implementation.
Feel free to write more scripts to help manage your codebase.
"""
import grid2op.Backend
import grid2op.Backend.pandaPowerBackend
import numpy as np
import tqdm
import os
import grid2op
from grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from typing import Literal, Tuple
from lightsim2grid.lightSimBackend import LightSimBackend
from jsonargparse import auto_cli
from L2RPN.Agents import DDPG, DDPGParams, TD3, TD3HParams
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from L2RPN.ReplayBuffer.VanillaReplayBuffer import ReplayBuffer
from pathlib import Path
import numpy as np
from numpy.core.multiarray import array as array
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import adamw
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass
from copy import deepcopy
from grid2op.Observation import BaseObservation

def run(env_name: str = r"E:\202504\2140prjE\l2rpn_case14_storage_\l2rpn_case14_storage_train", agent:Literal['DDPG','TD3']="DDPG",
        n_active:int=5000, replay_size:int=10000, rho_threshold:float=0.95, stage:Literal["TRAIN","VALIDATE","TEST"]="TRAIN", 
        batch_size:int=32, seed:int=0, verbose:bool=False) -> Tuple[float]:
    """
    Run one Reinforcement Learning (RL) loop. 

    Args:
        env_name (str, optional): Name of Grid2Op Environment to use, will be downloaded automatically (if it exists). Defaults to "l2rpn_case14_sandbox".
        agent (str, Literal['DDPG','TD3','SAC'], optional): Name of agent to use. Defaults to "DDPG".
        n_active (int, optional): Number of active steps to run for. Defaults to 5000.
        replay_size (int, optional): How big the replay buffer is. Defaults to 10000.
        rho_threshold (float, optional): Fraction of line thermal limit above which agent will be activated. Defaults to 0.95.
        stage (Literal['TRAIN','VALIDATE','TEST'], optional): Which stage we are in. Defaults to "TRAIN".
        batch_size (int, optional). Number of experiences in each batch. Defaults to 32.
        seed (int, optional): Seed for reproducibility. Defaults to 0.
        verbose (bool, optional): Whether to print out extra information. Defaults to False.

    Returns:
        Tuple[float]: _description_
    """
    # TODO: Figure out how to split into TRAIN, VALIDATE, and TEST Environments
    # We need to do this to prevent data from leaking, the agent should be evaluated on unseen data!

    # Temporary part of testing existence of dataset
    print("Dataset Existence:", os.path.exists(env_name))
    try:
        raw_env = grid2op.make(
            dataset=env_name,
            backend=PandaPowerBackend(),
            allow_detachment=True
        )
        print("Environment initialized successfully!")
        obs = raw_env.reset()
       
        print("Empty Load Success:", obs)

    except Exception as e:
        print("NO, Environment initialization failed.")
        print("   Reason:", e)
        raise

    env = TemplateEnvWrapper(env_name,
                             backend=PandaPowerBackend(),
                             rho_threshold=rho_threshold,
                             verbose=verbose,
                             env_kwargs={"allow_detachment":True})
    
    n_eps, ep_ids = env.get_env_size()
    # TODO: Figure out what Obervation / Action size is appropriate
    # Hint: You will need to implement this inside TemplateEnvWrapper

    def get_env_dims(env: TemplateEnvWrapper):
        obs_vec, _, _, _ = env.reset()
        obs_dim = obs_vec.shape[0]
        act_dim = env.env.n_storage  # env.env is grid2op.Environment
        return obs_dim, act_dim


    OBS_DIM, ACT_DIM = get_env_dims(env)
    ACT_LIMIT = 1 # Is it suits when we set 3?,can we reach the limit?
    # TODO: Agent not converging? Maybe try non-default hyperparameters / hyperparameter optimization

    hyperparameters = {}
    agent_lookup = {"DDPG":(DDPG, DDPGParams),
                    "TD3":(TD3,TD3HParams)}
    agent_class, agent_hparams = agent_lookup[agent]
    agent: DDPG | TD3 = agent_class(agent_hparams(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        act_limit=ACT_LIMIT,
        pi_hidden_sizes=[256, 256],
        q_hidden_sizes=[256, 256],
        activation_function=nn.Sigmoid,
        optim_pi=optim.Adam,
        optim_q=optim.Adam,
        lr_pi=1e-3,
        lr_q=1e-3,
        device='cuda',
        gamma=0.99,
        tau=0.005,
        noise_scale=0.1,
        n_steps=10
    ))
    #agent:DDPG|TD3 = agent_class(agent_hparams(obs_dim=OBS_DIM, act_dim=ACT_DIM, act_limit=ACT_LIMIT,**hyperparameters))

    # >> Replay Buffer <<
    buffer = ReplayBuffer(max_size=replay_size, obs_dim=OBS_DIM,
                          gamma=agent.gamma, N_steps=agent.n_steps)
    
    ep_no = 0
    total_steps = 0

    ep_rewards = []

    total_episodes = 0
    num_survived = 0
    num_failed = 0
    max_steps_survived = 0


    while total_steps < n_active:
        ep_pos = ep_no % n_eps
        ep_id = ep_ids[ep_pos]
        print(f"\nStarting Episode {ep_no} | Scenario ID: {ep_id}")

        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        
        print(f"Initial Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")

        done = terminated or truncated
       
        ep_steps, ep_reward = 0, info["reward"]

        while not done:
            action = agent.act(obs_vec)
            print(f"Step {ep_steps} | Action: {action}")
            print(f"In action: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")
            next_obs_vec, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if stage.upper() == "TRAIN":
                buffer.add(obs_vec, action, reward, next_obs_vec, terminated)
                if buffer.size > batch_size:
                    batch = buffer.sample(batch_size)
                    agent.update(batch)

            ep_reward += reward
            ep_steps += 1

            if terminated:
                num_failed += 1
                print(f"Agent failed early at step {ep_steps} in Scenario {ep_id}.")
                print(f"terminated: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")
            elif truncated:
                num_survived += 1
                print(f"Agent survived full duration in Scenario {ep_id}.")
                print(f"truncated: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")
            if ep_steps > max_steps_survived:
                max_steps_survived = ep_steps
            
            obs_vec = next_obs_vec

        ep_rewards.append(ep_reward)
        total_steps += ep_steps
        ep_no += 1
        total_episodes += 1
        print(f"Episode {ep_no} finished | Total Steps Survived: {ep_steps} | Total Reward: {ep_reward:.2f}")
        print("\n=== Training Summary ===")
        print(f"Total Episodes Run: {total_episodes}")
        print(f"Survived Episodes: {num_survived}")
        print(f"Failed Episodes: {num_failed}")
        print(f"Maximum Steps Survived: {max_steps_survived}")
        print("=============================\n")
    
    ep_rewards[ep_no] = ep_reward
    
    return np.mean(ep_rewards)


if __name__ == "__main__":
    auto_cli(run)


#            storage_charge = env.tracker.state.storage_charge
#            if storage_charge[0] == 0 or storage_charge[1] == 0:
#                penalty = -5.0  
#                reward += penalty
#                print(f"⚠️  Battery SOCs are both 0. Applying penalty: {penalty}")