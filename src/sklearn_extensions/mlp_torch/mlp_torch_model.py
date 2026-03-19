import copy
import torch
import torch.nn as nn
import torch.nn.functional  as F
import torch.utils
import torch.distributions
import torch.optim as optim
import torch_numopt
import numpy as np
from functools import reduce
import sklearn

class MLPModelTorch(nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes: list = None,
        activation='relu',
        last_layer='linear',
        dropout_rate=0,
        device='cpu'
    ):
        super().__init__()

        self.input_size = input_size
        if type(input_size) is int:
            flat_size = input_size
        else:
            flat_size = reduce(lambda x, y: x*y, input_size)

        self.layers = []
        if layer_sizes is None:
            layer_sizes = [20,15]
        self.layer_sizes = list(layer_sizes)

        self.layers = nn.ModuleList(
            [
                nn.Linear(size_in, size_out, device=device)
                for size_in, size_out in zip([flat_size] + self.layer_sizes, self.layer_sizes + [1])
            ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in self.layer_sizes]
        )
        
        match activation:
            case 'sigmoid':
                activation = nn.Sigmoid()
            case 'tanh':
                activation = nn.Tanh()
            case 'linear':
                activation = lambda x: x
            case 'relu':
                activation = nn.ReLU()
            case 'abs':
                activation = torch.abs
            case func if callable(func):
                pass
            case _:
                raise Exception("Use 'sigmoid', 'tanh', 'linear', 'relu', 'abs' or a lambda function.")
        self.activation = activation

        match last_layer:
            case 'sigmoid':
                last_layer = nn.Sigmoid()
            case 'tanh':
                last_layer = nn.Tanh()
            case 'linear':
                last_layer = lambda x: x
            case 'relu':
                last_layer = nn.ReLU()
            case 'abs':
                last_layer = torch.abs
            case func if callable(func):
                pass
            case _:
                raise Exception("Use 'sigmoid', 'tanh', 'linear', 'relu', 'abs' or a lambda function.")
        self.last_layer = last_layer
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        # Apply all hidden layers (no output layer)
        for dropout_layer, layer in zip(self.dropouts, self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = dropout_layer(x)
        
        # Apply output layer
        x = self.layers[-1](x)
        x = self.last_layer(x)

        return x
