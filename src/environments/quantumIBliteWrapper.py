'''
Wrapper for Siemens IBlite Reinforcement Learning Environment.
'''

import gym
from src.environments.IB_lite.IBGym import IBGym
import numpy as np

"""
        Initializes the underlying environment, seeds numpy and initializes action / observation spaces
        as well as other necessary variables

        :param setpoint:                determines behavior of industrial benchmark
        :param reward_type:             classic / delta - determines whether absolute or change in reward is returned
        :param action_type:             discrete / continuous - determines whether or not to use discretised action space (3^2 = 9) or continuous.
        :param observation_type:        classic / include_past - determines wether single or N state frames used as observation
        :param reset_after_timesteps:   how many timesteps can the environment run without resetting
        :param init_seed:               seed for numpy to make environment behavior reproducible
        :param n_past_timesteps:        if observation type is include_past, this determines how many state frames are used
        :param n_random_steps:          take given number of random steps after reset. (No prestepping if 0)
        :param scaling                  arctan / clipping / none - type of scaling applied to observation in state vector.
        :param clipping_value:          values used for clipping observations if clipping is used. [velocity, gain, fatigue, consumption] (Default values found by driving environment to max and min value)
"""
class QuantumIBLiteWrapper(gym.Wrapper):
    def __init__(self, setpoint=70, 
                reward_type="classic", 
                action_type="continuous", 
                observation_type="classic",
                reset_after_timesteps=1000, 
                init_seed=None, 
                n_past_timesteps=30, 
                n_random_steps=0, 
                scaling="none",
                clipping_value=[[0,100], [0,100], [0,300], [0,2750]]):

        # Init IBlite environment
        self.env = IBGym(setpoint, reward_type, action_type, observation_type, reset_after_timesteps,
                        init_seed, n_past_timesteps)
        super().__init__(self.env)
        
        self.n_random_steps = n_random_steps

        if self.n_random_steps > 0:
            # Buffer to log the initially taken random steps
            self.state_log = np.zeros((n_random_steps, *self.env.observation_space.shape))
            self.reward_log = np.zeros(n_random_steps)
            self.action_log = np.zeros(n_random_steps, dtype=int)

        if scaling == "arctan": self.scaling = self.arctan_scaling
        elif scaling == "clipping": self.scaling = self.clipping
        elif scaling == "none": self.scaling = lambda obs: obs
        else:
            raise ValueError("Scaling: {0} is invalid. Either arctan, clipping or none.".format(scaling))
        
        self.clipping_value = np.array(clipping_value)
        self._max_episode_steps = reset_after_timesteps

    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, environment_param:dict):
        return cls(environment_param["setpoint"], environment_param["reward_type"], environment_param["action_type"], 
                   environment_param["observation_type"], environment_param["reset_after_timesteps"], environment_param["init_seed"], 
                   environment_param["n_past_timesteps"], environment_param["n_random_steps"], environment_param["scaling"], 
                   environment_param["clipping_value"])

    def arctan_scaling(self, obs):
        obs[0] = obs[0] * 2 * np.pi / 100
        obs[1] = obs[1] * 2 * np.pi / 100
        obs[2] = np.arctan(obs[2]) + np.pi/2
        obs[3] = np.arctan(obs[3]) + np.pi/2

        return obs

    def clipping(self, obs):
        obs = np.clip(obs, a_min=self.clipping_value[:,0], a_max=self.clipping_value[:,1])
        # (obs - min)/max * 2pi
        # TODO fix since this does not take the max after subtracting min
        obs = (obs -self.clipping_value[:,0])/self.clipping_value[:,1] * 2 * np.pi

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self.scaling(obs)
        # 5 return values expected
        return obs, reward, done, False, info

    def reset(self, seed = None, options = None):
        if self.n_random_steps > 0:
            # Random stepping
            # IB does not support seed and options in reset, it uses the seed set in einvironment init
            # Reset only returns observation and no info
            _ = self.env.reset()

            for i in range(self.n_random_steps):
                # Sample random action from action space (continuous)
                self.action_log[i] = self.env.action_space.sample()
                # Step environment with random action and log returned state
                #obs, self.reward_log[i], _, _ = self.env.step(self.action_log[i])

                obs, self.reward_log[i], _, _, _ = self.step(self.action_log[i])
                self.state_log[i] = obs

            # Reset number of steps for timeout
            self.env.env_steps = 0

        else:
            # No random stepping
            obs = self.env.reset()
            obs = self.scaling(obs)

        # Collector expects two return values.
        return obs, None