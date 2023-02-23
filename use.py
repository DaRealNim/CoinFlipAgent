import gym
import numpy as np

from stable_baselines3 import DQN
from environment import CoinFlipEnv
import sys

# Initialize CoinFlipEnv
env = CoinFlipEnv()

# load agent
agent = DQN('MlpPolicy', env, verbose=1, device='cuda')
print("Loading checkpoint {}...".format(sys.argv[1]))
agent = agent.load(f"model/{sys.argv[1]}")
agent.set_env(env)

while True:
    state = input("Enter state: ").split()
    state = np.array([int(x) for x in state])
    action, _states = agent.predict(state, deterministic=True)
    print(action)