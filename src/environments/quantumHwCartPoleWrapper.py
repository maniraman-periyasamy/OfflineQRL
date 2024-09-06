import numpy as np
import gym

from src.environments.HwCartPole.qlinda_cart_pole.cart_pole_gym import QlindaCartPoleEnv
from src.environments.HwCartPole.qlinda_cart_pole import cart_pole_gym

POS_MODEL_PATH = r"./src/environments/HwCartPole/pos11Delta_flip"
THETA_MODEL_PATH = r"./src/environments/HwCartPole/theta11Delta_flip"

class QuantumHwCartPoleWrapper(gym.Wrapper):
    def __init__(self, 
                 n_past_obs = 9, 
                 actions = [-1, 0, 1], 
                 max_episode_steps = 500, 
                 action_type = "discrete", 
                 include_action = False, 
                 include_deltas = False, 
                 mult_pos_theta = False, 
                 mult_deltas = False,
                 scaling = False,
                 random_offset = False):
        """
        Wrapper for the hardware cart pole surrogate model.
        Args:
        n_past_obs          Number of past observations to include into the state. (Max value 9)
        max_episode_steps   Number of enivronment steps till termination. #TODO termination not yet included
        include_action      Flag to include performed action into the returned state.
        include_deltas      Flag to include position and theta delta into the returned state.
        mult_pos_theta      Flag to include a third entry in state vector which contains pos*theta
        mult_deltas         Flag to include a third entry in state vector which contains pos_delta * theta_delta
        scaling             Flag to scale observations to [-pi/2, pi/2].
        random_offset       Flag wether poles starts at origin or random offset at each offset.
        """
        assert n_past_obs <= 9 and n_past_obs >=0 

        self.env = QlindaCartPoleEnv(POS_MODEL_PATH, THETA_MODEL_PATH)
        super().__init__(self.env)
        #self.env.reset()

        self.action_type = action_type

        # Defining the action space
        if self.action_type == 'discrete':  # Discrete action space
            self.action_space = gym.spaces.Discrete(len(actions))

            # A list of all possible discretized actions
            self.env_action = actions

        elif self.action_type == 'continuous':  # Continuous action space
            self.action_space = gym.spaces.Box(np.array([-1]), np.array([+1]))

        else:
            raise ValueError('Invalid action_type. action_space can either be "discrete" or "continuous"')


        self.n_past_obs = n_past_obs
        self.include_action = include_action
        self.include_deltas = include_deltas
        self.mult_pos_theta = mult_pos_theta
        self.mult_deltas = mult_deltas
        self.random_offset = random_offset

        self.state_dim = 2
        if mult_pos_theta:
            self.state_dim += 1
        if mult_deltas:
            self.state_dim += 1
        if self.include_action:
            self.state_dim += 1
        if self.include_deltas:
            self.state_dim += 2
        self.state_dim *= (self.n_past_obs +1)

        self._max_episode_steps = max_episode_steps

        self.scaling = scaling
    
    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, env_params:dict):
        return cls(
            n_past_obs = env_params["n_past_obs"], 
            actions = env_params["actions"],  
            max_episode_steps = env_params["max_episode_steps"], 
            action_type = env_params["action_type"], 
            include_action = env_params["include_action"], 
            include_deltas = env_params["include_deltas"],  
            mult_pos_theta = env_params["mult_pos_theta"], 
            mult_deltas = env_params["mult_deltas"], 
            scaling = env_params["scaling"],
            random_offset = env_params["random_offset"]
        )
    
    def _process_obs(self, obs):
        """ 
        Scales observations to [-pi, pi] and excludes/includes features in the state vector
        """
        # env.step() returns obs as a list with 50 entries i.e. 10 observations with 5 entries
        obs = obs[-(self.n_past_obs + 1)*5:]
        # Turn observation into array with shape n_past_obs +1, state_dim i.e. current obs plus n past ones
        obs = np.array(obs).reshape(self.n_past_obs +1, 5)
        new_obs = np.zeros((self.n_past_obs +1, int(self.state_dim/(self.n_past_obs +1))))

        ### Scaling [-pi/2, pi/2]
        
        # Position in [32, 612]
        mean = 322  # (32 + 612)/2
        range = 290 # (612 - 32)/2
        new_obs[:,0] = (obs[:,0] - mean) / range * np.pi/2

        # Theta in [-1.5, 1.5]
        range = 1.5
        new_obs[:,1] = obs[:,1] / range * np.pi/2

        # Index to track entries in state
        state_idx = 2
        if self.mult_pos_theta:
            # Product will be in range [-pi^2, pi^2]
            new_obs[:,state_idx] = (new_obs[:,0] * new_obs[:,1]) / np.pi
            state_idx += 1
        
        if self.mult_deltas:
            # range [-14.31, 13.25]
            mean = -0.53
            range = 27.56/2
            new_obs[:,state_idx] = (obs[:,3] * obs[:,4] - mean)/range * np.pi/2
            state_idx += 1
        
        if self.include_action:
            new_obs[:,state_idx] = obs[:,2]
            state_idx += 1

        if self.include_deltas:
            # Position Delta 
            mean = 4.5
            range = 48.5
            new_obs[:,state_idx] = (obs[:,3] - mean)/range* np.pi/2
            state_idx += 1

            # Theta Delta
            mean = -0.01
            range = 0.26
            new_obs[:,state_idx] = (obs[:,4] - mean)/range* np.pi/2
            state_idx += 1
        
        return new_obs.reshape(-1)

    # Overwrite Environment
    def step(self, action):
        obs, reward, _, _, info, done = self.env.step(self.env_action[action])
        
        if self.scaling:
            new_obs = self._process_obs(obs)
        else:
            new_obs = obs

        return new_obs, reward, done, False, info
    
    def reset(self, seed = None, options = None):
        
        if self.random_offset:
            offset = np.random.randint(-50, 50)
        else:
            offset = None
        
        obs = self.env.reset(offset)

        if self.scaling:
            new_obs = self._process_obs(obs)
        else:
            new_obs = obs
        # No information is returned
        return new_obs, None