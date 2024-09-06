# Multi layer perceptron 
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model

class Mlp(Model):
    # Neural network function approximator, outputs logits (i.e. raw output of last linear layer) and log softmax
    # input_dim: size of input vector
    # output_dim: size of output vector
    # num_layers: number of hidden layers (Default 1)
    # hidden_dim: number of neurons per layer (Default [128])
    # activation: activation function after each linear layer (Default ReLU)
    # output_function: function applyed to ouput of final layer (Defaul LogSoftmax)
    def __init__(self, input_dim, output_dim, num_layers = 1, hidden_dim = [128], activation=nn.ReLU(inplace=True)):
        super().__init__()

        layers = []
        
        size = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(size, hidden_dim[i]))
            layers.append(activation)
            # Number of input features to succeeding layer is number of outputs of current layer
            size = hidden_dim[i]

        # Output layer without activation, since model should output logits
        layers.append(nn.Linear(size, output_dim))

        self.model = nn.Sequential(*layers)
    
    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model_params:dict):
        return cls( input_dim = model_params["input_dim"],
                    output_dim = model_params["output_dim"],
                    num_layers = model_params["num_layers"],
                    hidden_dim = model_params["hidden_dim"]
                  )
    
"""
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        batch = x.shape[0]
        logits = self.net(x.view(batch, -1))

        # Return logits and softmax
        return logits, F.log_softmax(logits, dim=1)"""