import math
import numpy as np
import torch.nn as nn, torch.optim as optim
import gymnasium as gym
from L2RPN.ReplayBuffer.VanillaReplayBuffer import ReplayBuffer
from L2RPN.Agents.DDPG import DDPG, DDPGParams # Score of -0.44 in ~470 episodes
from L2RPN.Agents.TD3 import TD3, TD3HParams


AGENT = "DDPG" # DDPG #"SAC"
OBS_DIM = 3
ACT_DIM = 1
ACT_LIMIT = 2

USE_PER = False
PER_ALPHA = 0.25
BETA = 0.6 # Parameter for importance sampling
BETA_DECAY = 1e-3
LEARNING_DELAY = 40
ENTROPY_ALPHA = 0.2
LOG_STD_MIN = -20
LOG_STD_MAX = 2
POLICY_DELAY = 2
BATCH_SIZE = 64
BUFFER_LEN = 200000
GAMMA = 0.99

HIDDEN_PI = [400,400]
HIDDEN_Q = [400, 400]
LR_PI = 1e-5
LR_Q = 1e-4
TAU = 0.001
NOISE_SCALE = 0.1
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5

N_GAMES = 3000
START_STEPS = 10000
NOISE_DEC_FACTOR = 0.9999
NOISE_MIN = 0.005
ENV_NAME = "Pendulum-v1"#"LunarLander-v2" # "MountainCarContinuous-v0"#
ENV_KWARGS = {"continuous":True} if ENV_NAME == "LunarLander-v2" else { }# continuous = True for lander
N = 10
M = 2
G = 20
POLICY_LOSS_MODE = "min"
AUTO_ENTROPY = False
N_STEPS = 3
VALIDATE_EVERY = 25

if __name__ == "__main__":
    env = gym.make(ENV_NAME, **ENV_KWARGS)
    if AGENT == "DDPG":
        agent_params = DDPGParams(
            OBS_DIM, ACT_DIM, HIDDEN_PI, HIDDEN_Q, nn.ReLU, ACT_LIMIT, optim.AdamW, optim.AdamW,
            LR_PI, LR_Q, "cpu", GAMMA, TAU, NOISE_SCALE, 3
        )
        agent = DDPG(agent_params)
    elif AGENT == "TD3":
        agent_params = TD3HParams(
            OBS_DIM, ACT_DIM, ACT_LIMIT, HIDDEN_PI, HIDDEN_Q, nn.ReLU, optim.AdamW, optim.AdamW,
            LR_PI, LR_Q, "cpu", GAMMA, TAU, NOISE_SCALE, TARGET_NOISE, POLICY_DELAY, NOISE_CLIP,
            3
        )
        agent = TD3(agent_params)
    else:
        raise ValueError(f"Invalid agent {AGENT}")
    scores, ma100, mem_beta, val_scores, ma_val = [], [], [], [], []
    pi_losses, q_losses = [], []
    max_score = -math.inf
    

    buffer = ReplayBuffer(BUFFER_LEN, OBS_DIM, GAMMA, N_STEPS)
    tot_steps = []
    val_scores = [-math.inf]
    for i in range(N_GAMES):
        score = 0
        steps = 0
        done = False
        obs, _ = env.reset()

        while not done:
            steps += 1
            if sum(tot_steps) < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.act(obs)
                if isinstance(agent, DDPG):
                    agent.noise_scale = max(agent.noise_scale*NOISE_DEC_FACTOR, NOISE_MIN)


            next_obs, reward, terminated, truncated, info = env.step(action)

            score += reward

            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            if len(buffer) >= START_STEPS:
                batch = buffer.sample(BATCH_SIZE)
                pi_loss, q_loss = agent.update(batch)
    
                pi_losses.append(pi_loss)
                q_losses.append(q_loss)
        
        scores.append(score)
        if score > max_score: 
            max_score = score
        tot_steps.append(steps)
        if not i % VALIDATE_EVERY and len(buffer) >= START_STEPS:
            for _ in range(5):
                obs, _ = env.reset()
                val_score = 0
                done = False
                while not done:
                    action = agent.act(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    val_score += reward
                    obs = next_obs
                    done = terminated or truncated
                val_scores.append(val_score)

        best_val_score = max(val_scores)
        avg_score = np.mean(scores[-50:])
        print(f"""Episode: {i} :: Score: {round(score, 2)} :: Max score: {round(max_score, 2)} :: Average score: {round(avg_score, 2)} :: Best Val score: {round(best_val_score, 2)}::Latest Val score: {round(val_scores[-1], 2)} Ep. Steps: {steps} :: Tot. Steps: {sum(tot_steps)}""")# :: Act noise: {round(agent.noise_scale, 3)}""")
        # print(f"Average policy loss: {np.mean(pi_losses[-100:])} :: Average critic loss: {np.mean([q_losses[-100:]])}")

    


