"""
Main script for running RL Agent Training.
Complete the pseudocode here as part of your project implementation.
Feel free to write more scripts to help manage your codebase.
"""

import numpy as np
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
import pandas as pd
from grid2op.PlotGrid import PlotMatplot
import sys


def run(env_name: str = "/Users/yanzeyang/Desktop/Group5RL-eg2140/Group5RL-eg2140/l2rpn_case14_storage_/l2rpn_case14_storage_val", agent:Literal['DDPG','TD3']="DDPG",
        n_active:int=300000, replay_size:int=150000, rho_threshold:float=0, stage:Literal["TRAIN","VALIDATE","TEST"]="VALIDATE", 
        batch_size:int = 512, seed:int=0, verbose:bool=False) -> Tuple[float]:
    

    #import excel files(Survival steps for each situation without agent)
    train_filename = "scenario_records_without_agent_train.xlsx"
    validate_filename = "scenario_records_without_agent_val.xlsx"
    test_filename = "scenario_records_without_agent_test.xlsx"

    if os.path.exists(train_filename):
        default_train = pd.read_excel(train_filename)
        default_val = pd.read_excel(validate_filename)
        default_test = pd.read_excel(test_filename)
        print(f"\nLoaded train records from {train_filename}")
        print(f"\nLoaded validate records from {validate_filename}")
        print(f"\nLoaded test records from {test_filename}")

    else:
        print("\nNo records found")

    #Setting the Dictionary
    train_dict = dict(zip(default_train["ID"], default_train["Steps"])) if not default_train.empty else {}
    val_dict = dict(zip(default_val["ID"], default_val["Steps"])) if not default_val.empty else {}
    test_dict = dict(zip(default_test["ID"], default_test["Steps"])) if not default_test.empty else {}

    print("Dataset Existence:", os.path.exists(env_name))

    #Setting grid2op parameters
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
    
    #Prepare for drawing
    plot_helper = PlotMatplot(env.env.observation_space)

    #Get scenario related parameters
    n_eps, ep_ids = env.get_env_size() 
    
    #Automatic capturing of observation and output dimensions
    def get_env_dims(env: TemplateEnvWrapper):
        obs_vec, _, _, _ = env.reset()
        obs_dim = obs_vec.shape[0]
        act_dim = env.env.n_storage  
        return obs_dim, act_dim

    OBS_DIM, ACT_DIM = get_env_dims(env)

    #Setting the output range of agent's action
    ACT_LIMIT = 2 

    #Setting DDPG-related parameters
    agent_lookup = {"DDPG":(DDPG, DDPGParams),
                    "TD3":(TD3,TD3HParams)}
    agent_class, agent_hparams = agent_lookup[agent]
    agent: DDPG | TD3 = agent_class(agent_hparams(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        act_limit=ACT_LIMIT,
        pi_hidden_sizes=[256, 128],
        q_hidden_sizes=[256, 128],
        activation_function=nn.Tanh,
        optim_pi=optim.Adam,
        optim_q=optim.Adam,
        lr_pi=3e-5,
        lr_q=1e-4,
        device='cpu',
        gamma=0.9,
        tau=0.01,
        noise_scale=0.1,
        n_steps=5
    ))
    
    buffer = ReplayBuffer(max_size=replay_size, obs_dim=OBS_DIM,
                          gamma=agent.gamma, N_steps=agent.n_steps)
    
    #Zeroing some recording parameters before the start of training
    ep_rewards = []
    episode_records = []

    ep_no = 0
    total_steps = 0
    total_episodes = 0
    num_survived = 0
    num_failed = 0

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

    action0_num_charge = 0
    action0_num_discharge = 0
    action0_num_donothing = 0

    action1_num_charge = 0
    action1_num_discharge = 0
    action1_num_donothing = 0

    action0_num_charge_rate = 0
    action0_num_discharge_rate = 0
    action0_num_donothing_rate = 0

    action1_num_charge_rate = 0
    action1_num_discharge_rate = 0
    action1_num_donothing_rate = 0

    #The trained model can be loaded and retrained
    if stage.upper() in ["VALIDATE", "TEST", "TRAIN"]:
        #Changing the model also requires changing the agent network parameters
        agent.load_checkpoint(Path("."), suffix="_256_128")
    
    #Training episode: 550, Validate episode:117, Test episode: 107
    while total_episodes < 117 :
        #Noise reduction per step during training
        if agent.noise_scale > 0:
            agent.noise_scale += -0.0001
        else:
            agent.noise_scale = 0
            
        ep_pos = ep_no % n_eps
        ep_id = ep_ids[ep_pos]
        
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        
        done = terminated or truncated
        
        ep_steps, ep_reward = 0, info["reward"]

        while not done:
            #Zeroing the difference between with and without agent situation
            delta_steps = 0

            #Detecting rho before agent action
            previous_rho = env.tracker.state.rho.copy()

            #Import Default Scenario Survival Steps
            if stage.upper() == "TRAIN":
                previous_steps = train_dict.get(ep_id, None)
            
            if stage.upper() == "VALIDATE":
                previous_steps = val_dict.get(ep_id, None)
            
            if stage.upper() == "TEST":
                previous_steps = test_dict.get(ep_id, None)
            
            #Agent Action begins.
            action = agent.act(obs_vec)
            action = np.clip(action, -2.0, 2.0)
            next_obs_vec, reward, terminated, truncated, info = env.step(action)
            
            #Plotting grid conditions at specific steps in specific scenarios
            if stage.upper() == "TEST" and ep_id =='Scenario_9' and ep_steps == 3:
                raw_obs = env.env.get_obs()  
                fig = plot_helper.plot_obs(raw_obs)
                fig.suptitle("Episode 14 • Step 3")
                fig.savefig("ep609_step372.png")
            
            #Getting the rho after agent action
            current_rho = env.tracker.state.rho
            max_prev_rho = np.max(previous_rho)
            max_current_rho = np.max(current_rho)

            #Record the behaviour of the agent for each scenario
            if 0 > action[0] >=-1: #Battery[0] charge
                action0_num_charge += 1
            
            if 0 > action[1] >=-1: #Battery[1] charge
                action1_num_charge += 1
            
            if 1 >= action[0] > 0: #Battery[0] discharge
                action0_num_discharge += 1
            
            if 1 >= action[1] > 0: #Battery[1] discharge
                action1_num_discharge += 1
                
            if action[0] > 1 or action[0] < -1: #Battery[0] donothing
                action0_num_donothing += 1
            
            if action[1] > 1 or action[1] < -1: #Battery[1] donothing
                action1_num_donothing += 1

            #Output the action of the agent and the SOC of the battery for each scenario  
            #print(f"Step {ep_steps} | Action: {action}")
            #print(f"In action: Battery SOCs: Battery 0 = {env.tracker.state.storage_charge[0]}, Battery 1 = {env.tracker.state.storage_charge[1]}")

            done = terminated or truncated
            
            #Reward the agent if its actions lower the rho.
            if stage.upper() == "TRAIN":
                if max_current_rho < max_prev_rho and max_current_rho > 0:
                    if -1 <= action[0] <= 1 or -1 <= action[1] <= 1:
                        reward_rho = (max_prev_rho - max_current_rho) * 10
                        reward += reward_rho
            
            #Rewards and penalties during initial training
            '''
            # Penalty for Battery 0
            if storage_charge[0] < 0.1 and 0 < action[0] <= 1:
                reward += -0.5
                #print("  Battery 0 is empty and agent tries to discharge. Penalty applied.")

            if storage_charge[0] > 14.9 and -1 <= action[0] < 0:  
                reward += -0.5
                #print("  Battery 0 is full and agent tries to charge. Penalty applied.")

            # Penalty for Battery 1
            if storage_charge[1] < 0.1 and 0 < action[0] <= 1:
                reward += -0.5
                #print("  Battery 1 is empty and agent tries to discharge. Penalty applied.")

            if storage_charge[1] > 6.9 and -1 <= action[0] < 0:  
                reward += -0.5
                #print("  Battery 1 is full and agent tries to charge. Penalty applied.")
            
            # Penalty for high rho
            if np.max(env.tracker.state.rho >= 1.2):
                if (action[0] > 1 or action[0] < -1) and (action[1] >1 or action[1] < -1):
                    reward += -1
                    #print("  High line loading detected (rho ≥ 1.2). agent do nothing.Penalty applied.")

            if np.max(env.tracker.state.rho >= 1):
                if (action[0] > 1 or action[0] < -1) and (action[1] >1 or action[1] < -1):
                    reward += -1
                    #print("  High line loading detected (rho ≥ 1). agent do nothing.Penalty applied.")

            if np.max(env.tracker.state.rho >= 0.8):
                if (action[0] > 1 or action[0] < -1) and (action[1] >1 or action[1] < -1):
                    reward += -1
                    #print("  High line loading detected (rho ≥ 0.9). agent do nothing.Penalty applied.")
            '''

            if stage.upper() == "TRAIN":
                ep_reward += reward
            
            ep_steps += 1

            if terminated:
                #Calculate the actions of the agent for this scenario
                num_failed += 1
                #Calculate the ratio of charging actions for Battery 0
                if action0_num_charge > 0:
                    action0_num_charge_rate = action0_num_charge/ep_steps*100
                else:
                    action0_num_charge_rate = 0
                #Calculate the ratio of discharging actions for Battery 0
                if action0_num_discharge > 0:
                    action0_num_discharge_rate = action0_num_discharge/ep_steps*100
                else:
                    action0_num_discharge_rate = 0
                #Calculate the ratio of do nothing actions for Battery 0
                if action0_num_donothing > 0:
                    action0_num_donothing_rate = action0_num_donothing/ep_steps*100
                else:
                    action0_num_donothing_rate = 0
                #Calculate the ratio of charging actions for Battery 1
                if action1_num_charge > 0:
                    action1_num_charge_rate = action1_num_charge/ep_steps*100
                else:
                    action1_num_charge_rate = 0
                #Calculate the ratio of discharging actions for Battery 1
                if action1_num_discharge > 0:
                    action1_num_discharge_rate = action1_num_discharge/ep_steps*100
                else:
                    action1_num_discharge_rate = 0
                #Calculate the ratio of do nothing actions for Battery 1
                if action1_num_donothing > 0:
                    action1_num_donothing_rate = action1_num_donothing/ep_steps*100
                else:
                    action1_num_donothing_rate = 0
                
                #Determine if this scenario has changed compared to the default scenario
                if ep_steps > previous_steps:
                    delta_steps = ep_steps - previous_steps
                    print(f"Improved! {delta_steps}")
                    #Preparation for subsequent calculations
                    improved_steps += delta_steps
                    improved_steps100 += delta_steps
                    
                    #Give agent a reward for exceeding the default performance.
                    if stage.upper() == "TRAIN":
                        ep_reward += (ep_steps - previous_steps) *10

                    num_improved +=1
                    num_improved100 +=1

                elif ep_steps < previous_steps:
                    delta_steps = ep_steps - previous_steps
                    print(f"Regressed!{delta_steps}")

                    regressed_steps += delta_steps
                    regressed_steps100 += delta_steps

                    #Give agent a Penalty for degradation
                    if stage.upper() == "TRAIN":
                        ep_reward = 0
                        ep_reward += (ep_steps - previous_steps) *5

                    num_regressed +=1
                    num_regressed100 +=1

                else:
                    print(f"Equal")
                    if stage.upper() == "TRAIN":
                        ep_reward = 0 

                    num_equal +=1
                    num_equal100 +=1

                #Record information about the current scenario
                episode_records.append({"ID": ep_id,"Steps": ep_steps,"Charge[0]":action0_num_charge_rate,"Discharge[0]":action0_num_discharge_rate,"Donothing[0]":action0_num_donothing_rate,"Charge[1]":action1_num_charge_rate,"Discharge[1]":action1_num_discharge_rate,"Donothing[1]":action1_num_donothing_rate})
                print(f"Agent failed early at step {ep_steps} in Scenario {ep_id}.Reward = {ep_reward} ")
                
                #Zeroing Battery Actions Count
                action0_num_charge = 0
                action0_num_discharge = 0
                action0_num_donothing = 0
                action1_num_charge = 0
                action1_num_discharge = 0
                action1_num_donothing = 0
                #print(f"terminated: Battery SOCs: Battery 1 = {env.tracker.state.storage_charge[0]}, Battery 2 = {env.tracker.state.storage_charge[1]}")

            elif truncated:
                #Same logic as above
                num_survived += 1
                if action0_num_charge > 0:
                    action0_num_charge_rate = action0_num_charge/ep_steps*100
                else:
                    action0_num_charge_rate = 0

                if action0_num_discharge > 0:
                    action0_num_discharge_rate = action0_num_discharge/ep_steps*100
                else:
                    action0_num_discharge_rate = 0

                if action0_num_donothing > 0:
                    action0_num_donothing_rate = action0_num_donothing/ep_steps*100
                else:
                    action0_num_donothing_rate = 0

                if action1_num_charge > 0:
                    action1_num_charge_rate = action1_num_charge/ep_steps*100
                else:
                    action1_num_charge_rate = 0

                if action1_num_discharge > 0:
                    action1_num_discharge_rate = action1_num_discharge/ep_steps*100
                else:
                    action1_num_discharge_rate = 0

                if action1_num_donothing > 0:
                    action1_num_donothing_rate = action1_num_donothing/ep_steps*100
                else:
                    action1_num_donothing_rate = 0
                
                if ep_steps > previous_steps:
                    delta_steps = ep_steps - previous_steps
                    print(f"Improved! {delta_steps}")

                    improved_steps += delta_steps
                    improved_steps100 += delta_steps

                    if stage.upper() == "TRAIN":
                        ep_reward += (ep_steps - previous_steps) *10

                    num_improved +=1
                    num_improved100 +=1
                    
                else:
                    print(f"Equal")
                    if stage.upper() == "TRAIN":
                        ep_reward = 0
                    num_equal +=1
                    num_equal100 +=1
                    
                episode_records.append({"ID": ep_id,"Steps": ep_steps,"Charge[0]":action0_num_charge_rate,"Discharge[0]":action0_num_discharge_rate,"Donothing[0]":action0_num_donothing_rate,"Charge[1]":action1_num_charge_rate,"Discharge[1]":action1_num_discharge_rate,"Donothing[1]":action1_num_donothing_rate})
                print(f"Agent survived full duration in Scenario {ep_id}.Reward = {ep_reward} ")

                action0_num_charge = 0
                action0_num_discharge = 0
                action0_num_donothing = 0
                action1_num_charge = 0
                action1_num_discharge = 0
                action1_num_donothing = 0

            #Save this scenario experience
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
        
        #Automation scripts that can be used during training
        if stage.upper() == "TRAIN":
            if num_regressed > 25:
                os.execv(sys.executable, ['python'] + sys.argv)
        
        #Output a summary every 100 scenarios        
        if total_episodes % 100 == 0:
            #Calculate how many more steps agent can survive on average
            if num_improved == 0:
                average_improved_steps = 0
            else:
                average_improved_steps =  improved_steps/num_improved
            #Calculate how many more steps agent can survive on average per 100 scenarios
            if num_improved100 == 0:
                average_improved_steps100 = 0
            else:
                average_improved_steps100 =  improved_steps100/num_improved100
            #Calculate the average steps of agent degradation
            if num_regressed == 0:
                average_regressed_steps = 0
            else:
                average_regressed_steps = regressed_steps/num_regressed
            #Calculate how many steps are degraded by agent per 100 scenarios
            if num_regressed100 == 0:
                average_regressed_steps100 = 0
            else:
                average_regressed_steps100 = regressed_steps100/num_regressed100

            #Calculate how many scenarios can be improved by agent per 100 scenarios
            improved_rate = num_improved100
            #Calculate how many scenarios can be improved by agent (total)
            tot_improved_rate = num_improved / total_episodes *100
            #Calculate how many scenarios are degraded by agent per 100 scenarios
            regressed_rate = num_regressed100
            #Calculate how many scenarios are degraded by agent (total)
            tot_regressed_rate = num_regressed / total_episodes *100
            #Calculate how many scenario agents did not impact the grid per 100 scenarios
            equal_rate = num_equal100
            #Calculate how many scenario agents did not impact the grid (total)
            tot_equal_rate = num_equal / total_episodes *100

            print("=============================\n")
            print("\n=== Training Summary ===")
            print(f"Improved Rate(every 100 scenarios):{improved_rate}% ")
            print(f"Improved Rate(total):{tot_improved_rate}% ")
            print(f"Average improved steps(every 100 scenarios):{average_improved_steps100} ")
            print(f"Average improved steps(total):{average_improved_steps} ")
            print(f"Regressed Rate(every 100 scenarios):{regressed_rate}% ")
            print(f"Regressed Rate(total):{tot_regressed_rate}% ")
            print(f"Average regressed steps(every 100 scenarios):{average_regressed_steps100} ")
            print(f"Average regressed steps(total):{average_regressed_steps} ")
            print(f"Equal Rate(every 100 scenarios):{equal_rate}% ")
            print(f"Equal Rate(total):{tot_equal_rate}% ")
            print(f"Total Episodes Run: {total_episodes}")
            print(f"Survived Episodes: {num_survived}")
            print(f"Failed Episodes: {num_failed}")
            print("=============================\n")
            
            #Zero the record every 100 steps
            num_survived100 = 0
            num_improved100 = 0
            num_regressed100= 0
            num_equal100 = 0
            improved_steps100 = 0
            regressed_steps100= 0

            #The automation script used to restart the training every 100 steps
            if stage.upper() == "TRAIN":
                    if regressed_rate > 10:
                        os.execv(sys.executable, ['python'] + sys.argv)

    #Save recorded information to an Excel document
    if episode_records:
        df = pd.DataFrame(episode_records)
        filename = f"model_records_train.xlsx"
        df.to_excel(filename, index=False)
        print(f"\nScenario records saved to {filename}")

    #Automation scripts are used to save the agents that meet the requirements or restart the training
    if stage.upper() == "TRAIN":
        if improved_rate > 50 and regressed_rate < 10:
            agent.save_checkpoint(Path("."), suffix="_final")
        else:
            os.execv(sys.executable, ['python'] + sys.argv)
    return np.mean(ep_rewards)

if __name__ == "__main__":
    auto_cli(run)


