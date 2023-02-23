import gym
import numpy as np

class CoinFlipEnv(gym.Env):
    def __init__(self):
        # flip 1, flip 5, label as fair, label as cheater
        self.action_space = gym.spaces.Discrete(4)

        # number of heads, number of tails, number of flips left
        self.observation_space = gym.spaces.Box(np.array([0, 0, 0]), np.array([np.inf, np.inf, np.inf]), shape=(3,))

        self.reset()

    # returns true if heads, false if tails
    def _flip_fair(self):
        return np.random.random() < 0.5

    def _flip_cheater(self):
        return np.random.random() < 0.75

    def _do_flip(self, flipfunc):
        if flipfunc():
            self.num_heads += 1
        else:
            self.num_tails += 1
        self.num_flips_left -= 1

    def step(self, action):
        flipfunc = self._flip_cheater if self.cheater else self._flip_fair
        reward = 0

        if action == 0:
            self._do_flip(flipfunc)
        elif action == 1:
            for i in range(5):
                self._do_flip(flipfunc)
        
        else:
            if (action == 2 and self.cheater) or (action == 3 and not self.cheater):
                reward = -30
            else:
                reward = 15
            self.num_heads = 0
            self.num_tails = 0
            self.cheater = np.random.random() < 0.5

        self.num_flips_left = max(0, self.num_flips_left+reward)
        finished = (self.num_flips_left <= 0)
        obs = self._get_obs()
        if finished:
            self.reset()
        return obs, reward, finished, {}

    def reset(self):
        self.num_heads = 0
        self.num_tails = 0
        self.num_flips_left = 100
        self.cheater = np.random.random() < 0.5
        return self._get_obs()


    def render(self, mode='human'):
        print(f"State: {self.num_heads} heads, {self.num_tails} tails, {self.num_flips_left} flips left")
        print(f"Hidden state: {'cheater' if self.cheater else 'fair'}")
        pass

    def close(self):
        pass

    def _get_obs(self):
        # return {
        #     "heads" : self.num_heads,
        #     "tails" : self.num_tails,
        #     "flips_left" : self.num_flips_left
        # }
        return np.array([self.num_heads, self.num_tails, self.num_flips_left]).astype(dtype=np.float32)