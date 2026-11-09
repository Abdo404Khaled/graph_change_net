"""MSTGCN on our runs of 1 kHz epochs: upstream's released graph network, a rate-scaled FeatureNet.

Upstream's ``OfficialFeatureNet`` refuses anything but 3000-sample epochs.
:class:`FeatureNet` is the same network with its first-layer kernels, strides
and pools given as settings: with upstream's values (``native`` sizes) it is
upstream's network, layer for layer and with the same parameter names, which
:func:`check_equivalence` verifies; under our configuration the first kernels
and strides are scaled by the sampling-rate ratio (10 for 1 kHz), the change
DeepSleepNet's paper prescribes for these branches (Fs/2 and Fs/16, 4 Fs and
Fs/2), which keeps 256 features per channel. The graph network is upstream's
``ReleasedMSTGCN`` unchanged; :mod:`baselines.sleepgraph.ours` says what our
configuration changes around it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from baselines.sleepgraph.ours import causal_windows, upstream_models


class FeatureNet(nn.Module):
    """Upstream's two-branch MSTGCN FeatureNet with configurable first-layer sizes.

    Parameters
    ----------
    small, large : dict
        Per branch: ``kernel`` and ``stride`` of the first convolution and
        ``pool`` of the first max-pooling.
    """

    def __init__(self, small: dict, large: dict):
        super().__init__()
        self.small = nn.Sequential(
            nn.Conv1d(1, 32, small["kernel"], stride=small["stride"]), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(small["pool"], small["pool"]), nn.Dropout(0.5),
            nn.Conv1d(32, 64, 8, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 8, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 8, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(8, 8), nn.Flatten())
        self.large = nn.Sequential(
            nn.Conv1d(1, 64, large["kernel"], stride=large["stride"]), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(large["pool"], large["pool"]), nn.Dropout(0.5),
            nn.Conv1d(64, 64, 6, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 6, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 6, padding="same"), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4, 4), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features of every channel of every epoch.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, V, samples]``.

        Returns
        -------
        torch.Tensor
            ``[N, V, F]``; ``F`` is 256 at upstream's sizes and 100 Hz, and at
            the scaled sizes and 1 kHz.
        """
        batch, channels, samples = x.shape
        flattened = x.reshape(batch * channels, 1, samples)
        features = torch.cat((self.small(flattened), self.large(flattened)), dim=1)
        return features.reshape(batch, channels, -1)


class MSTGCNRuns(nn.Module):
    """MSTGCN that labels every epoch of a run from the 5-epoch window ending at it.

    Our trainer has no subject labels and optimises the classification loss
    only, so the domain-adversarial head is replaced by an identity: its
    output is never used, and its 27M parameters would only inflate the count.

    Parameters
    ----------
    n_channels : int
        Electrodes.
    feature_net : dict
        ``small`` and ``large`` first-layer sizes of :class:`FeatureNet`.
    window : int
        Epochs per window (5, upstream's).
    adjacency : np.ndarray
        ``[V, V]`` static distance graph.
    """

    def __init__(self, n_channels: int, feature_net: dict, window: int, adjacency: np.ndarray):
        super().__init__()
        mstgcn = upstream_models().mstgcn
        self.features = FeatureNet(feature_net["small"], feature_net["large"])
        polynomials = mstgcn.released_static_polynomials(adjacency, 3)
        self.backbone = mstgcn.ReleasedMSTGCN(n_channels, polynomials, 1, timesteps=window)
        self.backbone.domain_head = nn.Identity()
        self.classifier = self.backbone.class_head[-1]
        self.window = window
        self.last_graph: torch.Tensor | None = None
        self._learned: torch.Tensor | None = None
        self.backbone.block.graph_learn.register_forward_hook(self._keep_graph)

    def _keep_graph(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        """Keep the learned graph of the labelled (last) epoch of each window.

        Parameters
        ----------
        module : nn.Module
            ``GraphLearn``.
        inputs : tuple
            Its inputs.
        output : tuple
            ``(adjacency [N, T, V, V], loss_1, loss_2)``.
        """
        self._learned = output[0][:, -1].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of every epoch of every run.

        Parameters
        ----------
        x : torch.Tensor
            Runs ``[B, S, V, samples]``.

        Returns
        -------
        torch.Tensor
            ``[B, S, 5]``.
        """
        batch, steps = x.shape[:2]
        features = self.features(x.flatten(0, 1).float()).view(batch, steps, x.shape[2], -1)
        out = self.backbone(causal_windows(features, self.window))
        self.last_graph = self._learned
        return out["logits"].view(batch, steps, -1)


def check_equivalence(seed: int) -> float:
    """Largest output difference between :class:`FeatureNet` at upstream's sizes and upstream's.

    Parameters
    ----------
    seed : int
        Seed of the random input.

    Returns
    -------
    float
        Maximum absolute difference of the 256 features, both networks in
        evaluation mode with the same weights; 0.0 means identical.
    """
    official = upstream_models().mstgcn.OfficialFeatureNet(4).eval()
    ours = FeatureNet({"kernel": 50, "stride": 6, "pool": 16},
                      {"kernel": 400, "stride": 50, "pool": 8}).eval()
    missing = ours.load_state_dict({k: v for k, v in official.state_dict().items()
                                    if not k.startswith("classifier")})
    if missing.missing_keys or missing.unexpected_keys:
        raise ValueError(f"parameter names differ: {missing}")
    x = torch.randn(3, 4, 3000, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return float((official.get_features(x) - ours(x)).abs().max())
