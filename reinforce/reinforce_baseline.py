import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.optim import Adam

torch.manual_seed(3)

def network(baseline=False):
    if baseline:
        return nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    else:
        return nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )

def train(policy, baseline, optimizer_policy, optimizer_baseline, current_episode, gamma=1):
    returns = [r for _, _, r in current_episode]
    T = len(current_episode)
    G = []

    for t in range(T-1):
        k = t+1
        R_k = returns[k:]
        R_k.reverse()
        G.append(sum(gamma**(k-t-1)*returns[k] for k in range(t+1, T)))

    for t in range(T-1):
        s_t, a_t, _ = current_episode[t]
        pi = policy(s_t)[a_t]
        v = baseline(s_t)
        delta = G[t] - v
        loss_baseline = -delta*v
        loss_policy = -(gamma**t)*G[t]*delta*torch.log(pi)
        optimizer_policy.zero_grad() ; optimizer_baseline.zero_grad() 
        loss_policy.backward(retain_graph=True)  ; loss_baseline.backward()
        optimizer_policy.step() ; optimizer_baseline.step()

policy = network()
optimizer_policy = Adam(policy.parameters(), lr=1e-4)
baseline = network(baseline=True)
optimizer_baseline = Adam(baseline.parameters(), lr=1e-4)

episodes = 500
past_rewards = []

env = gym.make("CartPole-v1")

for episode in range(episodes):
    s_t, info = env.reset()
    terminated = truncated = False
    current_episode = []

    while not terminated and not truncated:
        s_t = torch.from_numpy(s_t)
        a_t = torch.multinomial(policy(s_t), 1).item()
        s_t_p_1, r_t, terminated, truncated, info = env.step(a_t)
        current_episode.append((s_t, a_t, r_t))
        s_t = s_t_p_1
    
    train(policy, baseline, optimizer_policy, optimizer_baseline, current_episode)
    past_rewards.append(sum([r for _, _, r in current_episode]))

    if episode % 100 == 0:
        print(f"Episode {episode} - Reward {past_rewards[-1]}")
        plt.figure()
        plt.plot(past_rewards)
        plt.plot(np.convolve(past_rewards, np.ones(10)/10, mode='valid'))
        plt.savefig('train_reinforce_baseline.png')
        plt.close()
env.close()