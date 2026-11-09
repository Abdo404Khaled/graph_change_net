from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from components import Dimensions, build_component
from config import Config

NOTEBOOK_PREFIXES = {
    "embed_from": "graph_learner.embed_from",
    "embed_to": "graph_learner.embed_to",
    "project.": "graph_learner.project.",
    "memory": "graph_learner.memory",
    "sgcn1.": "graph_conv.layer1.",
    "sgcn2.": "graph_conv.layer2.",
    "gcn1.": "graph_conv.layer1.",
    "gcn2.": "graph_conv.layer2.",
    "sequence_to_model.": "decoder.context.",
    "sequence.": "sequence.rnn.",
    "decoder_out.": "decoder.out.",
    "decoder.": "decoder.rnn.",
    "head.": "head.",
}


class GraphChangeNet(nn.Module):
    """Sleep staging from a stable electrode graph plus a per-segment change.

    The network is a fixed pipeline of six slots, each filled by the component
    the config names (see ``components.py``)::

        x -> segments -> graph_learner -> graph_conv -> position -> sequence -> head
                                                                  \\-> decoder (stage 1)

    The modules are built in the order the notebooks created their layers, so
    the same seed gives the same initial weights; ``context`` comes between
    the readout and the head, and its default (``none``) has no weights, so it
    draws no random numbers.

    Parameters
    ----------
    cfg : Config
        Run settings; ``cfg.model`` selects the components.
    n_channels : int
        Electrodes, which are the graph nodes.
    coordinates : np.ndarray
        Montage positions ``[C, 3]`` in channel order (``data.read_montage``).

    Raises
    ------
    ValueError
        If ``coordinates`` does not hold one row per electrode.
    """

    def __init__(self, cfg: Config, n_channels: int, coordinates: np.ndarray):
        super().__init__()
        if coordinates.shape != (n_channels, 3):
            raise ValueError(f"coordinates {coordinates.shape} do not match {n_channels} "
                             "electrodes")
        spec = cfg.model
        dims = Dimensions(n_channels=n_channels, n_segments=cfg.data.n_segments,
                          segment_len=cfg.data.segment_len, d_model=spec.d_model,
                          d_graph=spec.d_graph, n_classes=len(cfg.data.stages),
                          coordinates=tuple(tuple(float(v) for v in row) for row in coordinates))
        self.dims = dims
        self.graph_learner = build_component("graph_learner", spec.graph_learner, dims)
        self.graph_conv = build_component("graph_conv", spec.graph_conv, dims)
        self.position = build_component("position", spec.position, dims)
        self.sequence = build_component("sequence", spec.sequence, dims)
        width = self.sequence.output_dim
        self.decoder = build_component("decoder", spec.decoder, dims, context_dim=width)
        self.readout = build_component("readout", spec.readout, dims, in_dim=width)
        self.context = build_component("context", spec.context, dims,
                                       in_dim=self.readout.output_dim)
        self.head = build_component("head", spec.head, dims, in_dim=self.context.output_dim)

    def segments(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Cut windows into segments and hide the masked cells.

        Masking happens here, before anything reads the signal, so hidden
        samples cannot come back through the graph.

        Parameters
        ----------
        x : torch.Tensor
            Windows ``[B, C, T * L]``.
        mask : torch.Tensor, optional
            ``[B, T, C]``, true where hidden.

        Returns
        -------
        torch.Tensor
            Node features ``[B, T, C, L]``.
        """
        cells = x.unflatten(-1, (self.dims.n_segments, self.dims.segment_len)).transpose(1, 2)
        return cells if mask is None else cells * ~mask.unsqueeze(-1)

    def backbone(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Everything up to the node representations shared by both stages.

        Parameters
        ----------
        x : torch.Tensor
            Windows ``[B, C, samples]``.
        mask : torch.Tensor, optional
            Hiding mask, see :meth:`segments`.

        Returns
        -------
        hidden : torch.Tensor
            Node representations ``[B, T, C, d_model]``.
        delta : torch.Tensor
            The change ``[B, T, C, C]``.
        stable : torch.Tensor
            The stable graph ``[C, C]``.
        """
        cells = self.segments(x, mask)
        scores, delta, stable = self.graph_learner(cells)
        hidden = self.position(self.graph_conv(cells, scores))
        return hidden, delta, stable

    def forward(self, x: torch.Tensor, elapsed: torch.Tensor | None = None):
        """Classify windows from the readout of the sequence outputs.

        The readout is the last segment for every experiment before Exp 13.
        Runs of consecutive epochs (Exp 14b) go through the backbone and the
        readout one epoch at a time, and the ``context`` component reads the
        resulting vectors in order before the head.

        Parameters
        ----------
        x : torch.Tensor
            Windows ``[B, C, samples]``, or runs of epochs ``[B, N, C, samples]``.
        elapsed : torch.Tensor, optional
            Standardised elapsed time ``[B]`` of rolling windows; only a
            readout with ``elapsed_time`` reads it.

        Returns
        -------
        logits : torch.Tensor
            ``[B, n_classes]``, or ``[B, N, n_classes]`` for runs of epochs.
        delta : torch.Tensor
            The change ``[B, T, C, C]``, or ``[B * N, T, C, C]``.
        stable : torch.Tensor
            The stable graph ``[C, C]``.
        """
        runs = x.dim() == 4
        hidden, delta, stable = self.backbone(x.flatten(0, 1) if runs else x)
        summary = self.readout(self.sequence(hidden), elapsed)
        if runs:
            summary = summary.unflatten(0, x.shape[:2])
        return self.head(self.context(summary)), delta, stable

    def penalty(self) -> torch.Tensor | float:
        """Regularisation terms the components left from their last forward pass.

        A component that needs a loss term of its own, such as the KL of the
        learned segment graph, stores it in a ``penalty`` attribute; the
        others have none.

        Returns
        -------
        torch.Tensor or float
            Sum of the terms, or 0.0 when no component has one.
        """
        terms = [module.penalty for module in self.children()
                 if getattr(module, "penalty", None) is not None]
        return sum(terms) if terms else 0.0

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor):
        """Rebuild hidden waveforms through the same layers stage 2 uses.

        Parameters
        ----------
        x : torch.Tensor
            Windows ``[B, C, samples]``.
        mask : torch.Tensor
            ``[B, T, C]``, true where hidden.

        Returns
        -------
        rebuilt : torch.Tensor
            Reconstructed segments ``[B, T, C, L]``.
        delta : torch.Tensor
            The change ``[B, T, C, C]``.
        """
        hidden, delta, _ = self.backbone(x, mask)
        return self.decoder(hidden, self.sequence(hidden)), delta

    def pretrain_outputs(self, x: torch.Tensor, mask: torch.Tensor):
        """Everything the stage-1 losses read, from one masked forward pass.

        The same computation as :meth:`reconstruct`, also returning the
        layer-1 node representations and the sequence outputs, which the
        "predict the future" loss uses (Exp 13).

        Parameters
        ----------
        x : torch.Tensor
            Windows ``[B, C, samples]``.
        mask : torch.Tensor
            ``[B, T, C]``, true where hidden.

        Returns
        -------
        rebuilt : torch.Tensor
            Reconstructed segments ``[B, T, C, L]``.
        delta : torch.Tensor
            The change ``[B, T, C, C]``.
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.
        context : torch.Tensor
            Sequence outputs ``[B, T, H]``.
        """
        hidden, delta, _ = self.backbone(x, mask)
        context = self.sequence(hidden)
        return self.decoder(hidden, context), delta, hidden, context


def from_notebook_keys(state: dict) -> dict:
    """Rename the parameters of a notebook checkpoint to the component layout.

    Parameters
    ----------
    state : dict
        State dict saved by the experiment notebooks (flat names such as
        ``sgcn1.balanced.weight``).

    Returns
    -------
    dict
        The same tensors under component names (``graph_conv.layer1.balanced.weight``).

    Raises
    ------
    KeyError
        If a parameter name has no mapping.
    """
    renamed = {}
    for key, value in state.items():
        prefix = next((p for p in NOTEBOOK_PREFIXES if key == p or key.startswith(p)), None)
        if prefix is None:
            raise KeyError(f"no component mapping for notebook parameter {key!r}")
        renamed[NOTEBOOK_PREFIXES[prefix] + key[len(prefix):]] = value
    return renamed


def load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load a checkpoint saved by ``run.py`` or by the experiment notebooks.

    Parameters
    ----------
    model : nn.Module
        Model to fill in place.
    path : Path
        Checkpoint file.
    device : torch.device
        Device to map the tensors to.
    """
    state = torch.load(path, map_location=device)
    if "embed_from" in state:
        state = from_notebook_keys(state)
    model.load_state_dict(state)


FRESH_PREFIXES = ("readout.", "context.", "head.")


def load_pretrained_parts(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load the stage-1 weights of another experiment, leaving the classifier parts fresh.

    Stage 1 trains everything except ``readout``, ``context`` and ``head``,
    so those keep this model's own initialisation and may differ in shape
    from the other experiment's (``pretrain.init_run``).

    Parameters
    ----------
    model : nn.Module
        Model to fill in place.
    path : Path
        ``pretrain_best.pt`` of the other run.
    device : torch.device
        Device to map the tensors to.

    Raises
    ------
    KeyError
        If the pretrained parts of the two models do not have the same parameters.
    """
    state = torch.load(path, map_location=device)
    kept = {key: value for key, value in state.items() if not key.startswith(FRESH_PREFIXES)}
    own = {key for key in model.state_dict() if not key.startswith(FRESH_PREFIXES)}
    if set(kept) != own:
        raise KeyError(f"{path} does not match this model: missing {sorted(own - set(kept))}, "
                       f"unexpected {sorted(set(kept) - own)}")
    model.load_state_dict(kept, strict=False)
