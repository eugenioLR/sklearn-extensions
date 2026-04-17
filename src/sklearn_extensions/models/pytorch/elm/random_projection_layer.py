import math
from torch import nn, optim


class RandomProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, randomizer=None, random_parameters=None, randomizer_bias=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        for p in self.linear_layer.parameters():
            p.requires_grad = False

        self.randomizer = randomizer
        self.random_parameters = random_parameters
        self.randomizer_bias = randomizer_bias
        self.reset_parameters()

    def reset_parameters(self):
        assert self.randomizer in {"uniform", "normal", "xavier", "xavier-normal", "kaiming", "kaiming-normal", None}
        assert self.randomizer_bias in {"uniform", "normal", "constant", None}

        match self.randomizer:
            case "uniform":
                if self.random_parameters is None:
                    self.random_parameters = {"a": -0.1, "b": 0.1}

                nn.init.uniform_(self.linear_layer.weight, **self.random_parameters)
            case "normal":
                if self.random_parameters is None:
                    self.random_parameters = {"mean": 0, "std": 0.01}

                nn.init.normal_(self.linear_layer.weight, **self.random_parameters)
            case "xavier":
                if self.random_parameters is None:
                    self.random_parameters = {}

                nn.init.xavier_uniform_(self.linear_layer.weight, **self.random_parameters)
            case "xavier-normal":
                if self.random_parameters is None:
                    self.random_parameters = {}

                nn.init.xavier_normal_(self.linear_layer.weight, **self.random_parameters)
            case "kaiming":
                if self.random_parameters is None:
                    self.random_parameters = {"a": math.sqrt(5)}  # math is used in pytorch too idk

                nn.init.kaiming_uniform_(self.linear_layer.weight, **self.random_parameters)
            case "kaiming-normal":
                if self.random_parameters is None:
                    self.random_parameters = {"a": math.sqrt(5)}  # math is used in pytorch too idk

                nn.init.kaiming_normal_(self.linear_layer.weight, **self.random_parameters)
            case None:
                pass
            case _:
                raise ValueError("Incorrect randomizer")

        if self.bias:
            match self.randomizer_bias:
                case "uniform":
                    nn.init.uniform_(self.linear_layer.bias, a=-0.01, b=0.01)
                case "normal":
                    nn.init.normal_(self.linear_layer.bias, mean=0, std=0.01)
                case "constant":
                    nn.init.constant_(self.linear_layer.bias, val=0.01)
                case None:
                    pass
                case _:
                    raise ValueError("Incorrect randomizer")

    def forward(self, x):
        return self.linear_layer(x)
