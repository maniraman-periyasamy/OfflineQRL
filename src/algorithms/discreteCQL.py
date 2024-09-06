""" Based on: https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/qlearning/torch/cql_impl.py
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .policy import Policy

class discreteCQL(Policy):
    def __init__(self, 
                 model,
                 optimizer,
                 num_actions,
                 state_dim,
                 device="cpu",
                 alpha=4.0,
                 discount=0.99,
                 polyak_target_update=False,
                target_update_frequency=8e3,
                tau=0.005,
                initial_eps = 0.1,
                end_eps = 0.1,
                eps_decay_period = 1,
                eval_eps=0,
     ) -> None:
        super().__init__()

        self.device = device

        # Set Q-network and imitator
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

        # Trade-off between conservative and Q-loss
        self.alpha = alpha

    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model:torch.nn, optimizer:torch.optim, num_actions:int, state_dim:int, policy_params:dict):
        return cls(
            model = model,
            optimizer = optimizer,
            num_actions = num_actions,
            state_dim = state_dim,
            alpha = policy_params["alpha"],
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
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

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
            # Double Q-Learning
            next_action = self.Q(state)[0].argmax(1, keepdim=True)
            q, _ = self.Q_target(next_state)

            target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        q, _ = self.Q(state)
        # Q value of action from batch
        current_Q = q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Compute Conservative loss
        logsumexp = torch.logsumexp(q, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(action.view(-1), num_classes=self.num_actions)
        data_values = (q * one_hot).sum(dim=1, keepdim=True)

        conservative_loss = (logsumexp - data_values).mean()

        loss = q_loss + self.alpha * conservative_loss

        # Optimize parameter
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        return loss.detach()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        # TODO to resume training copying Q_target might not be correct.
        self.Q.load_state_dict(torch.load(filename + "_Q", map_location=torch.device(self.device)))
        self.Q_target = copy.deepcopy(self.Q)
        if self.optimizer:
            self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))