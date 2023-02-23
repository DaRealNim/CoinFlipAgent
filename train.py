import gym
import numpy as np
import sys
import os

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from environment import CoinFlipEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

# Initialize CoinFlipEnv
env = CoinFlipEnv()

eval_env = CoinFlipEnv()

monitor = Monitor(eval_env)

eval_callback = EvalCallback(
    monitor,
    best_model_save_path="./model/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./model/",
    # name_prefix="dqn_coinflip",
    name_prefix="ppo_coinflip",
)

callback = CallbackList([checkpoint_callback, eval_callback])


# Initialize DQNAgent
# model = DQN(
#     "MlpPolicy",
#     env,
#     device="cuda",
#     verbose=1,
#     tensorboard_log="tb_logs/coinflip/",
#     exploration_fraction=0.15,
# )

model = PPO(
    "MlpPolicy",
    env,
    device="cuda",
    verbose=1,
    tensorboard_log="tb_logs/coinflip/",
)

# load agent if specified
if len(sys.argv) > 1:
    if sys.argv[1] == "best":
        print("Loading best checkpoint...")
        model = model.load(
            f"model/best_model.zip", print_system_info=True
        )
        model.set_env(env)
    else:
        print("Loading checkpoint {}...".format(sys.argv[1]))
        model = model.load(
            f"model/dqn_coinflip_{sys.argv[1]}_steps", print_system_info=True
        )
        model.set_env(env)

# NUM_TIMESTEPS = 1000000
# NUMBER_OF_CHECKPOINTS = 25

# get model current timestep
# current_timestep = model.num_timesteps

# for i in range(NUMBER_OF_CHECKPOINTS):

NUM_TIMESTEPS = 2000000

model.learn(
    total_timesteps=NUM_TIMESTEPS,
    log_interval=400,
    progress_bar=True,
    reset_num_timesteps=False,
    callback=callback,
)

# save model
# model.save(f"model/dqn_coinflip-{model.num_timesteps}.checkpoint")


# print(
#     "Saving checkpoint {}...".format(
#         current_timestep + (i + 1) * (NUM_TIMESTEPS // NUMBER_OF_CHECKPOINTS)
#     )
# )
# model.save(
#     f"model/dqn_coinflip-{current_timestep + (i+1)*(NUM_TIMESTEPS//NUMBER_OF_CHECKPOINTS)}.checkpoint"
# )
# model.save_replay_buffer(
#     f"model/dqn_coinflip-{current_timestep + (i+1)*(NUM_TIMESTEPS//NUMBER_OF_CHECKPOINTS)}.replay_buffer"
# )

os.system('curl -H "Priority: urgent" -d "Training complete" ntfy.sh/darealnim_bruh')
