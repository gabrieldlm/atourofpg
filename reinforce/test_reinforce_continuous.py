import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.optim import Adam

torch.manual_seed(3)
def network():
        return nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.ln = nn.LayerNorm(2)
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, 1)
        self.std = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.ln(x)
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        mean = self.mean(x)
        std = nn.functional.relu(self.std(x))
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        return torch.normal(mean, std)

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
        s_t, _, _ = current_episode[t]
        pi = policy.sample(s_t)
        v = baseline(s_t)
        delta = G[t] - v
        loss_baseline = -delta*v
        loss_policy = -(gamma**t)*G[t]*delta*torch.log(pi)
        optimizer_policy.zero_grad() , optimizer_baseline.zero_grad()
        loss_policy.backward(retain_graph=True) ; loss_baseline.backward()
        optimizer_policy.step() ; optimizer_baseline.step()

policy = Policy().to('cuda')
baseline = network().to('cuda')

optimizer_policy = Adam(policy.parameters(), lr=1e-4)
optimizer_baseline = Adam(baseline.parameters(), lr=1e-4)

episodes = 1000
past_rewards = []

env = gym.make("MountainCarContinuous-v0")
s_t, info = env.reset()

env.action_space

for episode in range(episodes):
    s_t, info = env.reset()
    terminated = truncated = False
    current_episode = []

    while not terminated and not truncated:
        s_t = torch.from_numpy(s_t).to('cuda')
        a_t = policy.sample(s_t).item()
        s_t_p_1, r_t, terminated, truncated, info = env.step([a_t])
        current_episode.append((s_t, a_t, r_t))
        s_t = s_t_p_1
    
    train(policy, baseline, optimizer_policy, optimizer_baseline, current_episode, gamma=0.99)
    past_rewards.append(sum([r for _, _, r in current_episode]))

    if episode % 100 == 0:
        print(f"Episode {episode} - Reward {past_rewards[-1]}")
        plt.figure()
        plt.plot(past_rewards)
        plt.plot(np.convolve(past_rewards, np.ones(10)/10, mode='valid'))
        plt.savefig('train_reinforce_continuous.png')
        plt.close()
env.close()