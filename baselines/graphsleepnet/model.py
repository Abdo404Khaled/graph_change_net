"""GraphSleepNet on our runs of 1 kHz epochs: upstream's released backbone, our input.

The backbone is sleep-graph-benchmark's ``ReleasedGraphSleepNetBackbone``
unchanged; :mod:`baselines.sleepgraph.ours` says what our configuration
changes around it. The same wrapper serves ``graphsleepnet_distance``, whose
graph is a fixed distance adjacency passed through upstream's own
``adjacency_override``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from baselines.sleepgraph.ours import DifferentialEntropy, causal_windows, upstream_models


class GraphSleepNetRuns(nn.Module):
    """GraphSleepNet that labels every epoch of a run from the window ending at it.

    Parameters
    ----------
    n_channels : int
        Electrodes, the graph's nodes.
    sampling_rate : float
        Hz of the epochs.
    window : int
        Epochs per window (5, upstream's).
    distance : np.ndarray or None
        A fixed ``[V, V]`` adjacency used instead of the learned graph, or
        None for GraphSleepNet's learned graph.
    """

    def __init__(self, n_channels: int, sampling_rate: float, window: int,
                 distance: np.ndarray | None):
        super().__init__()
        models = upstream_models()
        self.features = DifferentialEntropy(sampling_rate, n_channels)
        self.backbone = models.graphsleepnet.ReleasedGraphSleepNetBackbone(
            vertices=n_channels, timesteps=window)
        self.classifier = self.backbone.classifier
        self.window = window
        self.fixed = distance is not None
        self.register_buffer("distance", torch.as_tensor(
            distance if distance is not None else np.zeros((n_channels, n_channels)),
            dtype=torch.float32))
        self.last_graph: torch.Tensor | None = None

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
        windows = causal_windows(self.features(x), self.window)
        if self.fixed:
            graph = self.distance.expand(windows.shape[0], -1, -1)
        else:
            graph = self.backbone.blocks[0].graph_learn(windows[:, -1:])
        out = self.backbone(windows, adjacency_override=graph)
        self.last_graph = out["adjacency"].detach()
        return out["logits"].view(batch, steps, -1)
