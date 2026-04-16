from __future__ import annotations
from torch import optim, nn
import sklearn
from sklearn.metrics import accuracy_score, f1_score
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
        last_layer="logistic",
        loss_fn=None,
        dropout_rate=0,
        device="cpu",
        train_loop_fn=None,
        tol=1e-4,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
        info_freq=1000,
        random_state=None
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
            random_state=random_state
        )

    def fit(self, X, y):
        self.loss_fn_ = nn.CrossEntropyLoss() if self.loss_fn is None else self.loss_fn
        return super().fit(X, y)

    def score_report(self, X, y):
        pred = self.predict(X)
        return {
            "ACC": accuracy_score(y_true=y, y_pred=pred),
            "F1": f1_score(y_true=y, y_pred=pred),
        }
