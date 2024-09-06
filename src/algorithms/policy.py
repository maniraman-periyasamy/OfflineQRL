from abc import ABC, abstractmethod

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(ABC, nn.Module):
    def __init__(
        self, 
    ):
        super().__init__()

    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, policy_params:dict):
        raise NotImplementedError(f"fromDict() not implemented for {cls}.")
    
    def select_action(self, state, eval=False):
        raise NotImplementedError(f"select_action() not implemented for {self}.")

    def train_step(self, replay_buffer):
        raise NotImplementedError(f"train() not implemented for {self}.")

    def save(self, filename):
        raise NotImplementedError(f"save() not implemented for {self}.")

    def load(self, filename):
        raise NotImplementedError(f"load() not implemented for {self}.")