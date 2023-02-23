import gym
import numpy as np

from stable_baselines3.common.env_checker import check_env
from environment import CoinFlipEnv

# Initialize CoinFlipEnv
env = CoinFlipEnv()

check_env(env)