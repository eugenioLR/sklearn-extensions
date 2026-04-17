import torch
import scipy as sp
import numpy as np
from torch import nn
from sklearn.cluster import KMeans


class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, centers=None, widths=None, device="cpu"):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        if centers is None:
            centers = torch.empty((out_features, in_features), device=device)
            torch.nn.init.uniform_(centers)
        elif not isinstance(centers, torch.Tensor):
            centers = torch.as_tensor(centers, device=device)

        if widths is None:
            widths = torch.empty((out_features,), device=device)
            torch.nn.init.uniform_(widths)
        elif not isinstance(widths, torch.Tensor):
            widths = torch.as_tensor(widths, device=device)

        self.centers = torch.nn.Parameter(centers)
        self.widths = torch.nn.Parameter(widths)

    def initialize_clustering(
        self,
        X,
        n_samples=None,
        width_init="random",
        freeze_centers=True,
        freeze_widths=True,
    ):
        cluster_model = KMeans(n_clusters=self.out_features, n_init="auto")

        if n_samples is not None:
            indices = torch.randperm(X.shape[0])[:n_samples]
            X = X[indices]

        cluster_model = cluster_model.fit(X)
        self.centers.data = torch.Tensor(cluster_model.cluster_centers_, device=self.device)

        if width_init == "std":
            cluster_idx = cluster_model.predict(X)

            # Group points by cluster
            cluster_idx_sorted = cluster_idx.copy()
            cluster_idx_sorted.sort()
            X_sorted = X[cluster_idx.argsort()]
            X_grouped = np.split(X_sorted, np.unique(cluster_idx_sorted, return_index=True)[1][1:])
            for idx, X_cluster in enumerate(X_grouped):
                self.widths.data[idx] = 1 / X_cluster.var()

        elif width_init == "maxdist":
            cluster_idx = cluster_model.predict(X)

            # Group points by cluster
            cluster_idx_sorted = cluster_idx.copy()
            cluster_idx_sorted.sort()
            X_sorted = X[cluster_idx.argsort()]
            X_grouped = np.split(X_sorted, np.unique(cluster_idx_sorted, return_index=True)[1][1:])
            for idx, X_cluster in enumerate(X_grouped):
                distance_to_centroid = sp.spatial.distance.cdist(X_cluster, cluster_model.cluster_centers_[[idx], :])
                self.widths.data[idx] = np.sqrt(2 * self.out_features) / distance_to_centroid.max()

        elif width_init != "random":
            raise ValueError("Width initialization method not found. Try 'random', 'std' or 'maxdist'.")

        self.centers.requires_grad = not freeze_centers
        self.widths.requires_grad = not freeze_widths

    def forward(self, x):
        center_diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_center_diff = torch.square(center_diff).sum(axis=2)

        return torch.exp(-sq_center_diff * self.widths)
