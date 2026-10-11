"""Clustering model wrappers: K-Means / hierarchical / GMM with a common interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


@dataclass
class FittedModel:
    """Labels plus the centroids needed to assign *new* points without refitting."""
    name: str
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    estimator: Any = None  # KMeans / GMM expose predict; hierarchical uses centroids


def _centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([X[labels == c].mean(axis=0) for c in range(k)])


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42, n_init: int = 20) -> FittedModel:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return FittedModel("kmeans", k, km.labels_, km.cluster_centers_, km)


def fit_hierarchical(X: np.ndarray, k: int, link: str = "ward") -> FittedModel:
    ag = AgglomerativeClustering(n_clusters=k, linkage=link).fit(X)
    return FittedModel(f"hier_{link}", k, ag.labels_, _centroids(X, ag.labels_, k), ag)


def fit_gmm(X: np.ndarray, k: int, seed: int = 42, covariance_type: str = "diag") -> FittedModel:
    """Diagonal-covariance GMM. 'full' covariance is unreliable here because several
    one-hot / binary columns can become near-degenerate inside a component."""
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=5,
                         random_state=seed, reg_covar=1e-4).fit(X)
    lab = gm.predict(X)
    return FittedModel("gmm", k, lab, gm.means_, gm)


def fit_model(name: str, X: np.ndarray, k: int, seed: int = 42, n_init: int = 20) -> FittedModel:
    if name == "kmeans":
        return fit_kmeans(X, k, seed, n_init)
    if name == "gmm":
        return fit_gmm(X, k, seed)
    if name.startswith("hier_"):
        return fit_hierarchical(X, k, name.split("_", 1)[1])
    raise ValueError(f"Unknown model: {name}")


def assign_to_centroids(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid assignment and a softmax-style confidence.

    Confidence = softmax(-d^2 / (2*s^2)) where s is the median nearest-centroid distance;
    it is a *relative closeness* score, not a calibrated probability.
    """
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    s2 = np.median(d2.min(axis=1)) + 1e-9
    z = -d2 / (2 * s2)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return labels, p


def linkage_matrix(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """SciPy linkage for the dendrogram."""
    return scipy_linkage(X, method=method)


@dataclass
class SegmentModel:
    """Frozen segmentation model: centroids in preprocessed space, ordered by segment id.

    Scoring a new customer = preprocess with the saved pipeline, then nearest centroid.
    Nothing is refitted at scoring time.
    """
    algorithm: str
    k: int
    centroids: np.ndarray
    distance_scale: float  # median squared nearest-centroid distance on training data

    def predict(self, X: np.ndarray) -> np.ndarray:
        return assign_to_centroids(X, self.centroids)[0]

    def closeness_ratio(self, X: np.ndarray) -> np.ndarray:
        """Distance to the nearest centroid / distance to the second-nearest (in (0, 1]).

        Near 0 = clearly inside one segment; near 1 = on the boundary between two segments.
        """
        d = np.sqrt(((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        d.sort(axis=1)
        return d[:, 0] / np.maximum(d[:, 1], 1e-12)

    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(segment ids, relative-closeness matrix). Rows sum to 1; NOT a calibrated probability."""
        d2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        z = -d2 / (2 * self.distance_scale)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return d2.argmin(axis=1), p
