import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from environment import CoinFlipEnv
import sys

import seaborn as sns
import matplotlib.pyplot as plt

# Initialize CoinFlipEnv
env = CoinFlipEnv()

ACTION_NUMBER_TO_ACTION = {
    0: "Flip 1",
    1: "Flip 5",
    2: "Declare fair",
    3: "Declare cheater"
}

# load agent
# agent = DQN('MlpPolicy', env, verbose=1, device='cuda')
agent = PPO('MlpPolicy', env, verbose=1, device='cuda')
agent = agent.load(f"best_model_ppo.zip")


scores = []

N = 5000

for i in range(N):
    if (i % 10 == 0):
        print("Game {}".format(i))
    score = 0
    state = env.reset()
    while True:
        # print("+------------------+")
        # env.render()
        # print("Score = {}".format(score))

        action, _states = agent.predict(state, deterministic=True)
        # action = env.action_space.sample()

        # get value from single element array
        action = action.item()
        # print("Chosen action: {}".format(ACTION_NUMBER_TO_ACTION[action]))
        if (action == 2 and not env.cheater) or (action == 3 and env.cheater):
            score += 1
            # print("Correct!")
        # elif not (action == 0 or action == 1):
        #     print("Incorrect!")
        state, reward, done, info = env.step(action)
        if done:
            # print("Game over")
            break
        
        # press key to continue
        # input()
    scores.append(score)

print(f"Average score after {N} games: {np.mean(scores)}")
print(f"Median score after {N} games: {np.median(scores)}")
print(f"Max score after {N} games: {np.max(scores)}")
print(f"Min score after {N} games: {np.min(scores)}")
print(f"Standard deviation of scores after {N} games: {np.std(scores)}")

sns.histplot(scores)
plt.show()

# save scores
np.save("eval_scores-ppo.npy", scores)