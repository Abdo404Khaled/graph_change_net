"""Upstream's GraphSleepNet native training with a fixed distance graph instead of the learned one.

Run like ``python -m training.train`` (same arguments), in a process whose
``sys.path`` starts at the upstream repository. Upstream refuses
``--fixed-graph`` for the native condition, so this launcher keeps every
upstream step (preparation output, DE features, trainer, selection) and
changes only ``GraphSleepNetDE.forward``: the backbone receives upstream's own
distance graph (``models.mstgcn.load_or_generate_adjacency`` for the prepared
channels, the graph native MSTGCN uses) through upstream's
``adjacency_override``, the same argument its fixed-graph experiments use. The
graph is saved as ``<output-dir>/distance_adjacency.npy`` for the predictor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from models.graphsleepnet import GraphSleepNetDE
from models.mstgcn import load_or_generate_adjacency
from training import train


def option(argv: list[str], name: str) -> str:
    """The value that follows ``name`` on the command line.

    Parameters
    ----------
    argv : list of str
        Arguments.
    name : str
        Option, e.g. ``--data-dir``.

    Returns
    -------
    str
        Its value.
    """
    return argv[argv.index(name) + 1]


def main() -> None:
    """Patch the forward pass, save the graph, and hand over to upstream's ``training.train``."""
    argv = sys.argv[1:]
    first = sorted(Path(option(argv, "--data-dir")).glob("*.npz"))[0]
    channels = tuple(str(name) for name in np.load(first, allow_pickle=True)["ch_names"])
    adjacency, source = load_or_generate_adjacency(None, channels)
    output = Path(option(argv, "--output-dir"))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "distance_adjacency.npy", adjacency)
    print(f"fixed graph: {source}, {len(channels)} channels", flush=True)
    graph = torch.as_tensor(adjacency, dtype=torch.float32)

    def forward(self: GraphSleepNetDE, de: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream's forward pass with the fixed graph as the adjacency.

        Parameters
        ----------
        self : GraphSleepNetDE
            The model.
        de : torch.Tensor
            ``[B, T, V, 9]`` differential entropy.

        Returns
        -------
        dict
            Upstream's outputs.
        """
        normalized = (de - self.de_mean.to(de)) / self.de_std.to(de)
        return self.backbone(normalized,
                             adjacency_override=graph.to(de).expand(de.shape[0], -1, -1))

    GraphSleepNetDE.forward = forward
    train.main(argv)


if __name__ == "__main__":
    main()
