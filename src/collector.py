from .buffer import Buffer
import numpy as np
import copy

# Create a replay buffer 
# Read/write a replay buffer

class Collector:
    def __init__(self, env, buffer:Buffer = None, policy = None) -> None:
        """
        Args:
        env
        buffer
        policy
        """
        self.env = env
        self.buffer = buffer
        self.policy = policy

        self.reset()

    def reset(self, reset_buffer=True, gym_seed=None):
        self.data = {
            "state":[],
            "act":[],
            "rew":None,
            "terminated":False,
            "truncated":False,
            "done":False,
            "next_state":[],
            "info":None,
            #"policy":None
        }

        self.reset_env(gym_seed)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self):
        self.collect_step = 0
        self.collect_episode = 0
    
    def reset_buffer(self):
        if self.buffer is not None:
            self.buffer.reset()

    def reset_env(self, gym_seed=None):
        # TODO seed
        self.data["state"], self.data["info"] = self.env.reset()

    def collect(self, n_steps, random=False):
        """
        Args
        n_steps Number of steps to collect in buffer.
        random  Use random actions.
        """
        episode_start = True
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_reward_list = []
        episode_lens = []
        episode_idx = 0

        for step in range(n_steps):
            episode_timesteps += 1

            if (self.policy is None) or random:
                # Random policy
                self.data["action"] = self.env.action_space.sample()

            else:
                # Behaviour policy
                self.data["action"] = self.policy.select_action(self.data["state"])
            
            # TODO: noisy trajectories
            self.data["next_state"], self.data["reward"], self.data["terminated"], self.data["truncated"], self.data["info"] = self.env.step(self.data["action"])
            episode_reward += self.data["reward"]

            self.data["done"] = np.logical_or(self.data["terminated"], self.data["truncated"])

            # Only consider "done" if episode terminates due to failure condition
            done_float = float(self.data["done"]) if episode_timesteps < self.env._max_episode_steps else 0

            # Check with data should be buffered.
            if self.buffer != None:
                self.buffer.add(self.data["state"], self.data["action"], self.data["next_state"], self.data["reward"], done_float, self.data["done"], episode_start)

            self.data["state"] = copy.copy(self.data["next_state"])
            episode_start = False
            
            if self.data["done"]:
                # Reset environment
                # TODO Seeds
                # TODO returns nan when no episode was achieved.
                self.reset_env()
                self.data["done"] = False
                episode_start = True
                episode_reward_list.append(episode_reward)
                episode_reward = 0
                episode_lens.append(episode_timesteps)
                episode_timesteps = 0
                episode_num += 1
                #low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

        episode_reward_list = np.array(episode_reward_list)
        episode_lens = np.array(episode_lens)

        self.collect_step += episode_timesteps
        self.collect_episode += episode_num
        
        return {
            "n/ep": episode_num,
            "n/st": n_steps,
            "rews": episode_reward_list,
            "lens": episode_lens,
            "rew": episode_reward_list.mean() if len(episode_reward_list) >= 1 else 0,
            "len": episode_lens.mean() if len(episode_lens) >= 1 else 0,
            "rew_std": episode_reward_list.std() if len(episode_reward_list) >= 1 else 0,
            "len_std": episode_lens.std() if len(episode_lens) >= 1 else 0
        }

    def save_buffer(self, path: str):
        self.buffer.save(path)
        
    def load_buffer(self, path: str):
        self.buffer.load(path)