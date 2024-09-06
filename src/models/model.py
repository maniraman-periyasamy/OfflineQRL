"""
Base class to derive classical and quantum function approximators from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Each derived class must implement overwrite self.model with its own pytorch module.
        self.model = None

    def forward(self, x):
        # All derived classes will use this forward method.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        batch = x.shape[0]
        logits = self.model(x.view(batch, -1))

        # Return logits and softmax
        return logits, F.log_softmax(logits, dim=1)

    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model_params:dict):
        raise NotImplementedError(f"fromDict() not implemented for {cls}.")