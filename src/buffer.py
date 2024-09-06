import numpy as np
import torch

class Buffer(object):
	def __init__(self, 
			  state_dim, 
			  action_dim,
			  batch_size, 
			  buffer_size, 
			  device="cpu"):
		
		self.batch_size = batch_size
		self.state_dim = state_dim
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.crt_size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, action_dim))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

	def reset(self):
		self.ptr = 0
		self.crt_size = 0

		self.state[:,:] = 0
		self.action[:,:] = 0
		self.next_state[:,:] = 0
		self.reward[:,:] = 0
		self.not_done[:,:] = 0

	def add(self, state, action, next_state, reward, done, episode_done, episode_start):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
        # Ring buffer
		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)

	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save(self, save_folder):
		contrainer = { "state": 	 self.state[:self.crt_size],
					   "action": 	 self.action[:self.crt_size],
					   "next_state": self.next_state[:self.crt_size],
					   "reward":	 self.reward[:self.crt_size],
					   "not_done":   self.not_done[:self.crt_size],
					   "ptr": 		 self.ptr}
		
		np.savez(save_folder, **contrainer)

	def load(self, save_folder, buffer_size=None):
		""" 
		Args
		save_folder 	Path to .npz file containing buffer to load.
		"""
		npzfile = np.load(save_folder)
		reward_buffer = npzfile["reward"]
		
		assert reward_buffer.shape[0] <= self.max_size, "Buffer too large to load."
		self.crt_size = reward_buffer.shape[0]

		self.state[:self.crt_size] = npzfile["state"]
		self.action[:self.crt_size] = npzfile["action"]
		self.next_state[:self.crt_size] = npzfile["next_state"]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.not_done[:self.crt_size] = npzfile["not_done"]

		self.ptr = npzfile["ptr"]

		print(f"Replay Buffer loaded with {self.crt_size} elements.")