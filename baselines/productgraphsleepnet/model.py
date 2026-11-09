"""ProductGraphSleepNet on our runs of 1 kHz epochs: upstream's released backbone, our input.

The backbone is sleep-graph-benchmark's ``ReleasedProductGraphSleepNetBackbone``
unchanged (learned spatial graph over electrodes and temporal graph over the
window's epochs, Chebyshev convolution, bidirectional GRU, graph-wise
attention); :mod:`baselines.sleepgraph.ours` says what our configuration
changes around it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.sleepgraph.ours import DifferentialEntropy, causal_windows, upstream_models


class ProductGraphSleepNetRuns(nn.Module):
    """ProductGraphSleepNet that labels every epoch of a run from the window ending at it.

    The graph-wise attention module produces the logits directly; it is the
    classification head our trainer treats as ``classifier``.

    Parameters
    ----------
    n_channels : int
        Electrodes.
    sampling_rate : float
        Hz of the epochs.
    window : int
        Epochs per window (9, upstream's).
    backbone : dict
        ``gru_cell``, ``heads`` and ``dropout`` of upstream's backbone.
    """

    def __init__(self, n_channels: int, sampling_rate: float, window: int, backbone: dict):
        super().__init__()
        product = upstream_models().productgraphsleepnet
        self.features = DifferentialEntropy(sampling_rate, n_channels)
        self.backbone = product.ReleasedProductGraphSleepNetBackbone(
            n_channels, timesteps=window, gru_cell=backbone["gru_cell"], heads=backbone["heads"],
            dropout=backbone["dropout"], graph_alpha=0.0)
        self.classifier = self.backbone.gwat
        self.window = window
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
        out = self.backbone(causal_windows(self.features(x), self.window))
        self.last_graph = out["spatial_graph"].detach()
        return out["logits"].view(batch, steps, -1)
