import time
from copy import deepcopy
from organism import Organism
import gym
import torch
class Evaluator:
    def __init__(self, environment: gym.Env):
        self.env = environment

    def evaluate(self, organism: Organism, render=False, trials=10):
        total_reward = 0.0
        for t in range(trials):
            obs = torch.from_numpy(self.env.reset())
            obs = torch.tensor(obs, dtype=torch.float)
            done = False
            i = 0
            while not done and i < 1000:
                if render:
                    self.env.render()
                    time.sleep(0.03)
                action = organism.get_output(obs).data.numpy()
                obs, reward, done, info = self.env.step(action)
                obs = torch.from_numpy(obs)
                obs = torch.tensor(obs, dtype=torch.float)
                total_reward += reward / trials
                i += 1
        self.env.close()
        return total_reward
