""" Based on: https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/DQN.py
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .policy import Policy

class DQN(Policy):
    def __init__(
        self, 
        model,
        optimizer,
        num_actions,
        state_dim,
        device="cpu",
        discount=0.99,
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 1,
        end_eps = 0.001,
        eps_decay_period = 25e4,
        eval_eps=0.001,
    ):
        super().__init__()
        self.device = device

        # Determine network type
        self.Q = model
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = optimizer

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations, nn.Parameter to make it part of state_dict and restore from checkpoint.
        self.iterations = nn.Parameter(torch.zeros(1), requires_grad=False)

    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model:torch.nn, optimizer:torch.optim, num_actions:int, state_dim:int, policy_params:dict):
        return cls(
            model = model,
            optimizer = optimizer,
            num_actions = num_actions,
            state_dim = state_dim,
            discount = policy_params["discount"],
            polyak_target_update = policy_params["polyak_target_update"],
            target_update_frequency = policy_params["target_update_frequency"],
            tau = policy_params["tau"],
            initial_eps = policy_params["initial_eps"],
            end_eps = policy_params["end_eps"],
            eps_decay_period = policy_params["eps_decay_period"],
            eval_eps = policy_params["eval_eps"],
        )

    def select_action(self, state, eval=False):
        #eps = self.eval_eps if eval \
        #    else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Check wether model is in model.train() or model.eval()
        eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps) if self.training \
            else self.eval_eps

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                return int(self.Q(state)[0].argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train_step(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + done * self.discount * self.Q_target(next_state)[0].max(1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state)[0].gather(1, action)

        # Compute Q loss
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        return Q_loss.detach()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), str(filename + "_Q.pth"))
        torch.save(self.optimizer.state_dict(), str(filename + "_optimizer.pth"))

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q.pth", map_location=torch.device(self.device)))
        self.Q_target = copy.deepcopy(self.Q)
        # For inference we do not need optimizer.
        if self.optimizer:
            self.optimizer.load_state_dict(torch.load(filename + "_optimizer.pth"))