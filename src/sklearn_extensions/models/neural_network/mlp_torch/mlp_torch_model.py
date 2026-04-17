""" """

from __future__ import annotations
from typing import Iterable
from numbers import Integral, Real
from abc import ABC, abstractmethod
import copy
from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
import torch_numopt
import numpy as np
import sklearn
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, Options
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split


class MLPArchitectureTorch(nn.Module):
    """ """

    def __init__(
        self,
        input_size,
        output_size=1,
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
            layer_sizes = [100]
        self.layer_sizes = list(layer_sizes)

        self.layers = nn.ModuleList(
            [
                nn.Linear(size_in, size_out, device=device)
                for size_in, size_out in zip([flat_size] + self.layer_sizes, self.layer_sizes + [output_size])
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in self.layer_sizes])

        match activation:
            case "sigmoid" | "logistic":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case "linear" | "identity":
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
            case "sigmoid" | "logistic":
                last_layer = nn.Sigmoid()
            case "softmax":
                last_layer = nn.Softmax(1)
            case "log_softmax":
                last_layer = nn.LogSoftmax(1)
            case "tanh":
                last_layer = nn.Tanh()
            case "linear" | "identity":
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

    _parameter_constraints: dict = {
        "nn_model": [torch.nn.Module, None],
        "hidden_layer_sizes": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "activation": [StrOptions({"identity", "logistic", "tanh", "relu", "abs"}), callable],
        "optimizer_class": [
            StrOptions({"sgd", "adam", "newton", "lm"}),
            type,
            callable,
        ],
        "optimizer_params": [dict, None],
        "batch_size": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "last_layer": [StrOptions({"identity", "logistic", "tanh", "relu", "softmax", "log_softmax", "linear", "sigmoid"}), callable],
        "loss_fn": [torch.nn.Module, None],
        "dropout_rate": [Interval(Real, 0, 1, closed="left")],
        "device": [StrOptions({"cpu", "cuda", "mps", "xpu", "xla", "meta"}), Interval(Integral, 0, None, closed="left")],
        "train_loop_fn": [callable, None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],
        "n_iter_no_change": [
            Interval(Integral, 1, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "verbose": ["verbose"],
        "info_freq": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        nn_model,
        hidden_layer_sizes,
        activation,
        optimizer_class,
        optimizer_params,
        batch_size,
        n_iter,
        last_layer,
        loss_fn,
        dropout_rate,
        device,
        train_loop_fn,
        tol,
        validation_fraction,
        n_iter_no_change,
        verbose,
        info_freq,
        random_state,
    ):
        self.nn_model = nn_model
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.last_layer = last_layer
        self.loss_fn = loss_fn
        self.dropout_rate = dropout_rate
        self.device = device
        self.train_loop_fn = train_loop_fn
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.info_freq = info_freq
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, multi_output=True, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self._validate_params()

        if hasattr(self, "classes_"):
            self.multioutput_ = False
            self.n_features_out_ = len(self.classes_)
        else:
            self.multioutput_ = y.ndim != 1
            if not self.multioutput_:
                y = y[:, None]
            self.n_features_out_ = y.shape[1]

        rng = check_random_state(self.random_state)
        seed = rng.randint(0, 2**32 - 1)
        torch.manual_seed(seed)

        if not hasattr(self, "loss_fn_"):
            self.loss_fn_ = nn.MSELoss() if self.loss_fn is None else self.loss_fn

        self.batch_size_ = X.shape[0] if self.batch_size == "auto" else self.batch_size
        self.train_loop_fn_ = train_loop if self.train_loop_fn is None else self.train_loop_fn
        if self.nn_model is None:
            self.nn_model_ = MLPArchitectureTorch(
                input_size=self.n_features_in_,
                output_size=self.n_features_out_,
                layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                last_layer=self.last_layer,
                dropout_rate=self.dropout_rate,
                device=self.device,
            )
        else:
            self.nn_model_ = self.nn_model

        match self.optimizer_class:
            case "sgd":
                if self.optimizer_params is None:
                    self.optimizer_params_ = {"lr": 1e-5}
                else:
                    self.optimizer_params_ = self.optimizer_params

                self.optimizer_ = optim.SGD(self.nn_model_.parameters(), **self.optimizer_params_)
            case "adam":
                if self.optimizer_params is None:
                    self.optimizer_params_ = {"lr": 1e-3}
                else:
                    self.optimizer_params_ = self.optimizer_params

                self.optimizer_ = optim.Adam(self.nn_model_.parameters(), **self.optimizer_params_)
            case "newton":
                if self.optimizer_params is None:
                    self.optimizer_params_ = {
                        "lr_init": 1,
                        "lr_search_method": "backtrack",
                        "line_search_cond": "armijo",
                    }
                else:
                    self.optimizer_params_ = self.optimizer_params

                self.optimizer_ = torch_numopt.NewtonLS(self.nn_model_, **self.optimizer_params_)
            case "lm":
                if self.optimizer_params is None:
                    self.optimizer_params_ = {
                        "lr_init": 1,
                        "lr_search_method": "backtrack",
                        "line_search_cond": "armijo",
                    }
                else:
                    self.optimizer_params_ = self.optimizer_params

                self.optimizer_ = torch_numopt.LevenbergMarquardtLS(self.nn_model_, **self.optimizer_params_)
            case type():
                if issubclass(self.optimizer_class, optim.Optimizer):
                    if self.optimizer_params is None:
                        self.optimizer_params_ = {"lr": 1e-5}
                    else:
                        self.optimizer_params_ = self.optimizer_params

                    self.optimizer_ = self.optimizer_class(self.nn_model.parameters(), **self.optimizer_params_)
                elif issubclass(self.optimizer_class, torch_numopt.CustomOptimizer):
                    if self.optimizer_params is None:
                        self.optimizer_params_ = {
                            "lr_init": 1,
                            "lr_search_method": "backtrack",
                            "line_search_cond": "armijo",
                        }
                    else:
                        self.optimizer_params_ = self.optimizer_params

                    self.optimizer_ = self.optimizer_class(self.nn_model, **self.optimizer_params_)
            case _:
                raise ValueError("Expected `optimizer_class`'sgd', 'adam', 'newton' or a type.")

        self.history_ = self.train_loop_fn_(
            X,
            y,
            self.nn_model_,
            self.loss_fn_,
            self.optimizer_,
            self.n_iter,
            self.n_iter_no_change,
            self.tol,
            self.batch_size_,
            self.validation_fraction,
            self.device,
            self.info_freq,
            self.verbose,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = torch.tensor(check_array(X), device=self.device, dtype=torch.float32)
        self.nn_model_.eval()
        pred = self.nn_model_(X).detach().cpu().numpy()
        if not self.multioutput_:
            pred = pred.ravel()
        return pred


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
    X_val = torch.tensor(X_val, device=device, dtype=torch.float32)
    if np.issubdtype(y_train.dtype, np.integer):
        y_type = torch.long
    else:
        y_type = torch.float32
    y_train = torch.tensor(y_train, device=device, dtype=y_type)
    y_val = torch.tensor(y_val, device=device, dtype=y_type)

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
            loss = loss_fn(pred_batch, y_batch)
            train_loss += float(loss.detach()) * X_batch.shape[0]

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
            val_loss = float(loss_fn(pred_val, y_val))

        history["val_loss"].append(val_loss)

        # save best weights each epoch
        if val_loss < best_error - tol:
            best_error = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience = max_patience
        elif patience > 0:
            patience -= 1
        else:
            if verbose:
                print(f"Epoch {epoch+1:6d}/{n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_error:3.5f}")
            break

        if verbose and epoch % info_freq == 0:
            print(f"Epoch {epoch+1:6d}/{n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_error:3.5f}")

    nn_model.load_state_dict(best_weights)
    return history
