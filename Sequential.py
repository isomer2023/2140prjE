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
import pandas as pd




def run(env_name: str = "/Users/yanzeyang/Desktop/Group5RL-eg2140/Group5RL-eg2140/l2rpn_case14_storage_/l2rpn_case14_storage_train", agent:Literal['DDPG','TD3']="DDPG",
        n_active:int=300000, replay_size:int=10000, rho_threshold:float=0.4, stage:Literal["TRAIN","VALIDATE","TEST"]="TRAIN", 
        batch_size:int = 128, seed:int=0, verbose:bool=False) -> Tuple[float]:
    
    #import excel files
    train_filename = "scenario_records_train.xlsx"
    validate_filename = "scenario_records_val.xlsx"
    test_filename = "scenario_records_test.xlsx"



    if os.path.exists(train_filename):
        history_train = pd.read_excel(train_filename)
        history_val = pd.read_excel(validate_filename)
        history_test = pd.read_excel(test_filename)

        print(f"\nLoaded train records from {train_filename}")
        print(f"\nLoaded validate records from {validate_filename}")
        print(f"\nLoaded test records from {test_filename}")


    else:
        print("\nNo records found")

    #set dict
    train_dict = dict(zip(history_train["ID"], history_train["Steps"])) if not history_train.empty else {}
    val_dict = dict(zip(history_val["ID"], history_val["Steps"])) if not history_val.empty else {}
    test_dict = dict(zip(history_test["ID"], history_test["Steps"])) if not history_test.empty else {}




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
    

    def get_env_dims(env: TemplateEnvWrapper):
        obs_vec, _, _, _ = env.reset()
        obs_dim = obs_vec.shape[0]
        act_dim = env.env.n_storage  
        return obs_dim, act_dim


    OBS_DIM, ACT_DIM = get_env_dims(env)
    ACT_LIMIT = 2 

    hyperparameters = {}
    agent_lookup = {"DDPG":(DDPG, DDPGParams),
                    "TD3":(TD3,TD3HParams)}
    agent_class, agent_hparams = agent_lookup[agent]
    agent: DDPG | TD3 = agent_class(agent_hparams(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        act_limit=ACT_LIMIT,
        pi_hidden_sizes=[1024, 1024],
        q_hidden_sizes=[1024, 1024],
        activation_function=nn.Tanh,
        optim_pi=optim.Adam,
        optim_q=optim.Adam,
        lr_pi=1e-4,
        lr_q=1e-4,
        device='cpu',
        gamma=0.95,
        tau=0.005,
        noise_scale=0.3,
        n_steps=10
    ))
    

    buffer = ReplayBuffer(max_size=replay_size, obs_dim=OBS_DIM,
                          gamma=agent.gamma, N_steps=agent.n_steps)
    
    ep_no = 0
    total_steps = 0

    ep_rewards = []

    total_episodes = 0
    num_survived = 0
    num_failed = 0
    max_steps_survived = 0
    num_survived100 = 0
    num_10 = 0
    
    episode_records = []

    num_improved =0     
    num_regressed =0
    num_equal =0
    num_improved100 =0     
    num_regressed100 =0
    num_equal100 =0
    tot_improved_rate = 0

    improved_steps = 0
    improved_steps100 = 0

    regressed_steps = 0
    regressed_steps100 = 0




    while total_episodes < 5500:
    #while total_steps < n_active:
        if agent.noise_scale > 0:
            agent.noise_scale += -0.001
        else:
            agent.noise_scale = 0

        ep_pos = ep_no % n_eps
        ep_id = ep_ids[ep_pos]
        
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        
        done = terminated or truncated
        
        ep_steps, ep_reward = 0, info["reward"]

        base_reward = 0

        while not done:

            delta_steps = 0

            previous_rho = env.tracker.state.rho.copy()

            if stage.upper() == "TRAIN":
                previous_steps = train_dict.get(ep_id, None)
            
            if stage.upper() == "VALIDATE":
                previous_steps = val_dict.get(ep_id, None)
            
            if stage.upper() == "TEST":
                previous_steps = test_dict.get(ep_id, None)
            
            
            action = agent.act(obs_vec)
            action = np.clip(action, -2.0, 2.0)
            next_obs_vec, reward, terminated, truncated, info = env.step(action)

            #base_reward = ep_steps*0.01  
            #reward = base_reward

            current_rho = env.tracker.state.rho
            max_prev_rho = np.max(previous_rho)
            max_current_rho = np.max(current_rho)

            #print(f"Step {ep_steps} | Action: {action}")

            #print(f"In action: Battery SOCs: Battery 0 = {env.tracker.state.storage_charge[0]}, Battery 1 = {env.tracker.state.storage_charge[1]}")

            done = terminated or truncated

            storage_charge = env.tracker.state.storage_charge
            storage_Emax = env.tracker.state.storage_Emax
            
            
            if stage.upper() == "TRAIN":
                if max_current_rho < max_prev_rho and max_current_rho > 0:
                    if -1 <= action[0] <= 1 or -1 <= action[1] <= 1:
                        reward_rho = (max_prev_rho - max_current_rho) * 10
                        reward += reward_rho
            
            

            
            # 惩罚 Battery 0
            if storage_charge[0] < 0.1 and 0 < action[0] <= 1:
                reward += -0.2
                #print("  Battery 0 is empty and agent tries to discharge. Penalty applied.")

            if storage_charge[0] > 14.9 and -1 <= action[0] < 0:  
                reward += -0.2
                #print("  Battery 0 is full and agent tries to charge. Penalty applied.")

            # 惩罚 Battery 1
            if storage_charge[1] < 0.1 and 0 < action[0] <= 1:
                reward += -0.2
                #print("  Battery 1 is empty and agent tries to discharge. Penalty applied.")

            if storage_charge[1] > 6.9 and -1 <= action[0] < 0:  
                reward += -0.2
                #print("  Battery 1 is full and agent tries to charge. Penalty applied.")
            
           
            if np.max(env.tracker.state.rho >= 1.2):
                if (action[0] > 1 or action[0] < -1) and (action[1] >1 or action[1] < -1):
                    reward += -0.3
                    #print("  High line loading detected (rho ≥ 1.2). agent do nothing.Penalty applied.")


            if np.max(env.tracker.state.rho >= 1):
                if (action[0] > 1 or action[0] < -1) and (action[1] >1 or action[1] < -1):
                    reward += -0.3
                    #print("  High line loading detected (rho ≥ 1). agent do nothing.Penalty applied.")
            
            
            if stage.upper() == "TRAIN":
                if ep_steps - previous_steps ==1:
                    reward += 100
                
                ep_reward += reward
            
            ep_steps += 1


            if terminated:
                num_failed += 1
                
                if ep_steps > previous_steps:
                    delta_steps = ep_steps - previous_steps
                    print(f"Improved! {delta_steps}")

                    improved_steps += delta_steps
                    improved_steps100 += delta_steps

                    
                    if stage.upper() == "TRAIN":
                        
                        ep_reward += (ep_steps - previous_steps) *1
                    

                    num_improved +=1
                    num_improved100 +=1

                elif ep_steps < previous_steps:
                    delta_steps = ep_steps - previous_steps

                    regressed_steps += delta_steps
                    regressed_steps100 += delta_steps


                    print(f"Regressed!{delta_steps}")
                    
                    
                    if stage.upper() == "TRAIN":
                        
                        ep_reward += (ep_steps - previous_steps) *1
                    

                    num_regressed +=1
                    num_regressed100 +=1
                else:
                    print(f"Equal")
                    
                    num_equal +=1
                    num_equal100 +=1

                #episode_records.append({"ID": ep_id,"Steps": ep_steps})

                print(f"Agent failed early at step {ep_steps} in Scenario {ep_id}.Reward = {ep_reward} ")

                #print(f"terminated: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")
                
            elif truncated:
                num_survived += 1
                num_survived100 += 1
                
                if ep_steps > previous_steps:
                    delta_steps = ep_steps - previous_steps
                    print(f"Improved! {delta_steps}")

                    improved_steps += delta_steps
                    improved_steps100 += delta_steps

                    
                    
                    if stage.upper() == "TRAIN":
                        
                        ep_reward = (ep_steps - previous_steps) *20
                    

                    num_improved +=1
                    num_improved100 +=1
                    
                else:
                    print(f"Equal")
                    

                    num_equal +=1
                    num_equal100 +=1
                    

                #episode_records.append({"ID": ep_id,"Steps": ep_steps})

                print(f"Agent survived full duration in Scenario {ep_id}.Reward = {ep_reward} ")
                #print(f"truncated: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")
            if ep_steps > max_steps_survived:
                max_steps_survived = ep_steps
            
            if stage.upper() == "TRAIN":
                buffer.add(obs_vec, action, reward, next_obs_vec, done)
                if buffer.size > batch_size:
                    batch = buffer.sample(batch_size)
                    agent.update(batch)

            obs_vec = next_obs_vec

        ep_rewards.append(ep_reward)
        total_steps += ep_steps
        ep_no += 1
        total_episodes += 1

        if total_episodes % 100 == 0:
            survival_rate = num_survived100
            tot_survival_rate = num_survived / total_episodes *100


            average_improved_steps =  improved_steps/num_improved
            average_improved_steps100 =  improved_steps100/num_improved100

            average_regressed_steps = regressed_steps/num_regressed
            average_regressed_steps100 = regressed_steps100/num_regressed100


            improved_rate = num_improved100
            tot_improved_rate = num_improved / total_episodes *100

            regressed_rate = num_regressed100
            tot_regressed_rate = num_regressed / total_episodes *100

            equal_rate = num_equal100
            tot_equal_rate = num_equal / total_episodes *100

            #print(f"Episode {ep_no} finished | Total Steps Survived: {ep_steps} | Total Reward: {ep_reward:.2f}")
            print("=============================\n")
            print("\n=== Training Summary ===")
            print(f"Improved Rate(100):{improved_rate}% ")
            print(f"Improved Rate(tot):{tot_improved_rate}% ")
            print(f"Average improved steps(100):{average_improved_steps100} ")
            print(f"Average improved steps(tot):{average_improved_steps} ")
            print(f"Regressed Rate(100):{regressed_rate}% ")
            print(f"Regressed Rate(tot):{tot_regressed_rate}% ")
            print(f"Average regressed steps(100):{average_regressed_steps100} ")
            print(f"Average regressed steps(tot):{average_regressed_steps} ")
            print(f"Equal Rate(100):{equal_rate}% ")
            print(f"Equal Rate(tot):{tot_equal_rate}% ")
            print(f"Total Episodes Run: {total_episodes}")
            print(f"Survived Episodes: {num_survived}")
            print(f"Failed Episodes: {num_failed}")
            print("=============================\n")
            
            num_survived100 = 0
            num_improved100 = 0
            num_regressed100= 0
            num_equal100 = 0
            improved_steps100 = 0
            regressed_steps100= 0
        
        


        #if total_episodes == 550:
        #    tot_survival_rate = num_survived / total_episodes *100
        #    print("\n=== Training Summary ===")
        #    #print(f"------------------------->>> Survival Rate(100):{survival_rate}% <<<-------------------------")
        #    print(f"------------------------->>> Survival Rate(tot):{tot_survival_rate}% <<<-------------------------")
        #    print(f"Total Episodes Run: {total_episodes}")
        #    print(f"Survived Episodes: {num_survived}")
        #    print(f"Failed Episodes: {num_failed}")
        #    #print(f"Maximum Steps Survived: {max_steps_survived}")
        #    print("=============================\n")

    '''
    if episode_records:
        df = pd.DataFrame(episode_records)
        filename = f"scenario_records_test.xlsx"
        df.to_excel(filename, index=False)
        print(f"\nScenario records saved to {filename}")
    '''
    
    return np.mean(ep_rewards)




if __name__ == "__main__":
    auto_cli(run)


