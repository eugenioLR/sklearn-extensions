from __future__ import annotations
import numpy as np
import scipy as sp
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
import torch
from torch import optim, nn
from .mlp_torch_model import MLPModelTorch


class MLPClassifierTorch(MLPModelTorch, sklearn.base.ClassifierMixin):
    def __init__(
        self,
        nn_model=None,
        hidden_layer_sizes=(100,),
        activation="relu",
        optimizer_class="adam",
        optimizer_params=None,
        batch_size="auto",
        n_iter=100000,
        last_layer="linear",
        loss_fn=None,
        dropout_rate=0,
        device="cpu",
        train_loop_fn=None,
        tol=1e-4,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
        info_freq=1000,
        random_state=None,
    ):
        super().__init__(
            nn_model=nn_model,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            batch_size=batch_size,
            n_iter=n_iter,
            last_layer=last_layer,
            loss_fn=loss_fn,
            dropout_rate=dropout_rate,
            device=device,
            train_loop_fn=train_loop_fn,
            tol=tol,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=verbose,
            info_freq=info_freq,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.loss_fn_ = nn.NLLLoss() if self.loss_fn is None else self.loss_fn
        return super().fit(X, y)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = torch.tensor(check_array(X), device=self.device, dtype=torch.float32)
        self.nn_model_.eval()
        with torch.no_grad():
            proba = self.nn_model_(X).detach().cpu().numpy()

        proba = sp.special.softmax(proba, axis=1)
        return proba  # Keep shape (n_samples, n_classes)

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score_report(self, X, y):
        pred = self.predict(X)
        return {
            "ACC": accuracy_score(y_true=y, y_pred=pred),
            "F1": f1_score(y_true=y, y_pred=pred),
        }
