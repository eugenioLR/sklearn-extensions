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
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.metrics import *
from .mlp_torch_model import MLPModelTorch

class MLPClassifierTorch(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(
        self,
        input_size: int,
        nn_model = None,
        layer_sizes: list = None,
        optimizer_class=optim.Adam,
        optimizer_params=None,
        activation='relu',
        last_layer='linear',
        loss_fn=None,
        dropout_rate=0,
        device='cpu',
        patience = 20,
        val_size = 0.1,
        batch_size = 5000,
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
            nn_model = MLPModelTorch(
                input_size=input_size,
                layer_sizes=layer_sizes,
                activation=activation,
                last_layer=last_layer,
                dropout_rate=dropout_rate,
                device=device
            )
        self.nn_model = nn_model

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
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
                    optimizer_params = {"lr_init": 1, "lr_search_method": "backtrack", "line_search_cond": "armijo"}
                
                self.optimizer = torch_numopt.NewtonLS(self.nn_model, **optimizer_params)
            case "lm":
                if optimizer_params is None:
                    optimizer_params = {"lr_init": 1, "lr_search_method": "backtrack", "line_search_cond": "armijo"}
                
                self.optimizer = torch_numopt.LevenbergMarquardtLS(self.nn_model, **optimizer_params)
            case type():
                if issubclass(optimizer_class, optim.Optimizer): 
                    if optimizer_params is None:
                        optimizer_params = {"lr": 1e-5}

                    self.optimizer = optimizer_class(self.nn_model.parameters(), lr=1e-5)
                elif issubclass(optimizer_class, torch_numopt.CustomOptimizer):
                    if optimizer_params is None:
                        optimizer_params = {"lr_init": 1, "lr_search_method": "backtrack", "line_search_cond": "armijo"}

                    self.optimizer = optimizer_class(self.nn_model, **optimizer_params)
            case _:
                raise ValueError("Expected `optimizer_class`'sgd', 'adam', 'newton' or a type.")
        
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.patience = patience
        self.val_size = val_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.info_freq = info_freq
        self.fitted = False
    
    def __sklearn_is_fitted__(self):
        return self.fitted
    
    def predict(self, X):
        self.nn_model.eval()
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        return self.nn_model(X).detach().cpu().numpy()
    
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, shuffle=True)

        X_train = torch.tensor(X_train, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32)
        X_val = torch.tensor(X_val, device=self.device, dtype=torch.float32)
        y_val = torch.tensor(y_val, device=self.device, dtype=torch.float32)

        best_mse = np.inf
        best_weights = None
        patience = self.patience
        batch_start = torch.arange(0, len(X_train), self.batch_size)
        self.history = []
        for epoch in range(self.n_epochs):
            # Fit network on training set
            self.nn_model.train()
            train_loss = 0
            for start in batch_start:
                X_batch = X_train[start:start+self.batch_size]
                y_batch = y_train[start:start+self.batch_size]

                pred_batch = self.nn_model.forward(X_batch)
                loss = self.loss_fn(y_batch, pred_batch)
                train_loss += float(loss)

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, torch_numopt.CustomOptimizer):
                    self.optimizer.step(X_batch, y_batch, self.loss_fn)
                else:
                    self.optimizer.step()
            train_loss /= len(batch_start)

            # Evaluate on validation set
            self.nn_model.eval()

            pred_val = self.nn_model.forward(X_val)
            val_loss = float(self.loss_fn(y_val, pred_val))
            self.history.append(val_loss)

            # save best weights each epoch
            if val_loss < best_mse:
                best_mse = val_loss
                best_weights = copy.deepcopy(self.nn_model.state_dict())
                patience = self.patience
            elif patience > 0:
                patience -= 1
            else:
                print(f"Epoch {epoch+1:6d}/{self.n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_mse:3.5f}")
                break

            if self.verbose and epoch % self.info_freq == 0:
                print(f"Epoch {epoch+1:6d}/{self.n_epochs}: Train loss: {train_loss:3.5f}, Val loss: {val_loss:3.5f}, Best loss: {best_mse:3.5f}")
        
        self.nn_model.load_state_dict(best_weights)
        self.fitted = True

        return self

    def score_report(self, X, y):
        pred = self.predict(X)
        print(y)
        return {
            "ACC": accuracy_score(y_true = y, y_pred = pred),
            "F1": f1_score(y_true = y, y_pred = pred),
        }


if __name__ == "__main__":
    from sklearn.datasets import *
    X, y = make_regression(1000, n_features=10)
    y = y[:, None]
    print(X.shape)
    print(y.shape)
    model = MLPRegressorTorch(X.shape[1])
    model.fit(X, y)