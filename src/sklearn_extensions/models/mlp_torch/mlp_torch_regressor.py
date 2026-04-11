from __future__ import annotations
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from torch import optim, nn
from .mlp_torch_model import MLPModelTorch


class MLPRegressorTorch(MLPModelTorch, RegressorMixin):
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
        patience=20,
        val_size=0.1,
        batch_size=5000,
        n_epochs=100000,
        verbose=True,
        info_freq=1000,
    ):
        if loss_fn is None:
            loss_fn = nn.MSELoss()

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
            patience,
            val_size,
            batch_size,
            n_epochs,
            verbose,
            info_freq,
        )

    def score_report(self, X, y):
        pred = self.predict(X)
        print(y)
        return {
            "R2": r2_score(y_true=y, y_pred=pred),
            "RMSE": root_mean_squared_error(y_true=y, y_pred=pred),
            "MAE": mean_absolute_error(y_true=y, y_pred=pred),
        }
