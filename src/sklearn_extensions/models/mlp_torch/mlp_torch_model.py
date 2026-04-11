""" """

from __future__ import annotations
from typing import Iterable
from abc import ABC
import copy
from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
import torch_numopt
import numpy as np
import sklearn
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split


class MLPArchitectureTorch(nn.Module):
    """ """

    def __init__(
        self,
        input_size,
        layer_sizes: list = None,
        activation="relu",
        last_layer="linear",
        dropout_rate=0,
        device="cpu",
    ):
        super().__init__()

        self.input_size = input_size
        if isinstance(input_size, int):
            flat_size = input_size
        else:
            flat_size = reduce(lambda x, y: x * y, input_size)

        self.layers = []
        if layer_sizes is None:
            layer_sizes = [20, 15]
        self.layer_sizes = list(layer_sizes)

        self.layers = nn.ModuleList(
            [nn.Linear(size_in, size_out, device=device) for size_in, size_out in zip([flat_size] + self.layer_sizes, self.layer_sizes + [1])]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in self.layer_sizes])

        match activation:
            case "sigmoid":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case "linear":
                activation = lambda x: x
            case "relu":
                activation = nn.ReLU()
            case "abs":
                activation = torch.abs
            case func if callable(func):
                pass
            case _:
                raise ValueError("Use 'sigmoid', 'tanh', 'linear', 'relu', 'abs' or a lambda function.")
        self.activation = activation

        match last_layer:
            case "sigmoid":
                last_layer = nn.Sigmoid()
            case "tanh":
                last_layer = nn.Tanh()
            case "linear":
                last_layer = lambda x: x
            case "relu":
                last_layer = nn.ReLU()
            case "abs":
                last_layer = torch.abs
            case func if callable(func):
                pass
            case _:
                raise ValueError("Use 'sigmoid', 'tanh', 'linear', 'relu', 'abs' or a lambda function.")
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


class MLPModelTorch(ABC, sklearn.base.BaseEstimator):
    """ """

    def __init__(
        self,
        input_size: int,
        nn_model=None,
        layer_sizes: list = None,
        optimizer_class=optim.Adam,
        optimizer_params=None,
        activation="relu",
        last_layer="linear",
        loss_fn=None,
        dropout_rate=0,
        device="cpu",
        train_loop_fn=None,
        patience=20,
        val_size=0.1,
        batch_size=5000,
        n_epochs=100000,
        verbose=True,
        info_freq=1000,
    ):
        self.device = device
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.last_layer = last_layer
        self.dropout_rate = dropout_rate
        if nn_model is None:
            nn_model = MLPArchitectureTorch(
                input_size=input_size,
                layer_sizes=layer_sizes,
                activation=activation,
                last_layer=last_layer,
                dropout_rate=dropout_rate,
                device=device,
            )
        self.nn_model = nn_model

        if loss_fn is None:
            loss_fn = nn.MSELoss()
        self.loss_fn = loss_fn

        match optimizer_class:
            case "sgd":
                if optimizer_params is None:
                    optimizer_params = {"lr": 1e-5}

                self.optimizer = optim.SGD(self.nn_model.parameters(), **optimizer_params)
            case "adam":
                if optimizer_params is None:
                    optimizer_params = {"lr": 1e-5}

                self.optimizer = optim.Adam(self.nn_model.parameters(), **optimizer_params)
            case "newton":
                if optimizer_params is None:
                    optimizer_params = {
                        "lr_init": 1,
                        "lr_search_method": "backtrack",
                        "line_search_cond": "armijo",
                    }

                self.optimizer = torch_numopt.NewtonLS(self.nn_model, **optimizer_params)
            case "lm":
                if optimizer_params is None:
                    optimizer_params = {
                        "lr_init": 1,
                        "lr_search_method": "backtrack",
                        "line_search_cond": "armijo",
                    }

                self.optimizer = torch_numopt.LevenbergMarquardtLS(self.nn_model, **optimizer_params)
            case type():
                if issubclass(optimizer_class, optim.Optimizer):
                    if optimizer_params is None:
                        optimizer_params = {"lr": 1e-5}

                    self.optimizer = optimizer_class(self.nn_model.parameters(), lr=1e-5)
                elif issubclass(optimizer_class, torch_numopt.CustomOptimizer):
                    if optimizer_params is None:
                        optimizer_params = {
                            "lr_init": 1,
                            "lr_search_method": "backtrack",
                            "line_search_cond": "armijo",
                        }

                    self.optimizer = optimizer_class(self.nn_model, **optimizer_params)
            case _:
                raise ValueError("Expected `optimizer_class`'sgd', 'adam', 'newton' or a type.")

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if train_loop_fn is None:
            train_loop_fn = train_loop
        self.train_loop_fn = train_loop_fn
        self.patience = patience
        self.val_size = val_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.info_freq = info_freq
        self.history = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self.n_features_in_ = X.shape[1]

        self.history = self.train_loop_fn(
            X,
            y,
            self.nn_model,
            self.loss_fn,
            self.optimizer,
            self.n_epochs,
            self.patience,
            self.batch_size,
            self.val_size,
            self.device,
            self.info_freq,
            self.verbose,
        )
    
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self.nn_model.eval()
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        return self.nn_model(X).detach().cpu().numpy()

def train_loop(
    X,
    y,
    nn_model,
    loss_fn,
    optimizer,
    n_epochs=10000,
    max_patience=1000,
    tol=1e-6,
    batch_size=100000,
    val_size=0.1,
    device="cpu",
    info_freq=100,
    verbose=False,
) -> dict[str, Iterable]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=True)

    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
    X_val = torch.tensor(X_val, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val, device=device, dtype=torch.float32)

    nn_model = nn_model.to(device)

    best_error = np.inf
    best_weights = None
    batch_start = torch.arange(0, len(X_train), batch_size)
    patience = max_patience
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_train))
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]

        # Fit network on training set
        nn_model.train()
        train_loss = 0
        for start in batch_start:
            X_batch = X_train_shuf[start : start + batch_size]
            y_batch = y_train_shuf[start : start + batch_size]

            pred_batch = nn_model(X_batch)
            loss = loss_fn(y_batch, pred_batch)
            train_loss += float(loss) * X_batch.shape[0]

            loss.backward()
            if isinstance(optimizer, torch_numopt.CustomOptimizer):
                optimizer.step(X_batch, y_batch, loss_fn)
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loss /= X_train.shape[0]
        history["train_loss"].append(train_loss)

        # Evaluate on validation set
        nn_model.eval()

        with torch.no_grad():
            pred_val = nn_model(X_val)
            val_loss = float(loss_fn(y_val, pred_val))
        
        history["val_loss"].append(val_loss)

        # save best weights each epoch
        if val_loss < best_error - tol:
            best_error = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience = max_patience
        elif patience > 0:
            patience -= 1
        else:
            print(f"Epoch {epoch+1:6d}/{n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_error:3.5f}")
            break

        if verbose and epoch % info_freq == 0:
            print(f"Epoch {epoch+1:6d}/{n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_error:3.5f}")

    nn_model.load_state_dict(best_weights)
    return history
