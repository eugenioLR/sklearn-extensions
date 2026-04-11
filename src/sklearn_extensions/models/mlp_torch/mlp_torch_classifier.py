from __future__ import annotations
from torch import optim, nn
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from .mlp_torch_model import MLPModelTorch


class MLPClassifierTorch(MLPModelTorch, sklearn.base.ClassifierMixin):
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
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        super().__init__(
            input_size,
            nn_model,
            layer_sizes,
            optimizer_class,
            optimizer_params,
            activation,
            last_layer,
            loss_fn,
            dropout_rate,
            device,
            train_loop_fn,
            patience,
            val_size,
            batch_size,
            n_epochs,
            verbose,
            info_freq,
        )

    def score_report(self, X, y):
        pred = self.predict(X)
        return {
            "ACC": accuracy_score(y_true=y, y_pred=pred),
            "F1": f1_score(y_true=y, y_pred=pred),
        }
