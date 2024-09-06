import gym
import numpy as np

class QuantumCartPoleWrapper(gym.Wrapper):
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        super().__init__(self.env)
    
        self.env.reset()
        self._max_episode_steps = self.env._max_episode_steps
        self.state_dim = self.env.observation_space.shape[0] 
    
    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, env_params:dict):
        return cls()
        
    # Overwrite Environment
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[0] = (obs[0] / 4.8 + 1) * np.pi
        obs[1] = np.arctan(obs[1])
        obs[2] = ((obs[2] / ((24/180)*np.pi) + 1) * np.pi)
        obs[3] = np.arctan(obs[3])
        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        return self.env.reset(seed=seed)
