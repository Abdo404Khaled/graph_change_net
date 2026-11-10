from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS = ("graph_learner", "graph_conv", "position", "sequence", "decoder", "head", "readout",
         "context")
REGISTRY: dict[str, dict[str, type]] = {slot: {} for slot in SLOTS}


@dataclass(frozen=True)
class Dimensions:
    """Sizes every component may need, fixed by the data and the model widths.

    Parameters
    ----------
    n_channels : int
        Electrodes, which are the graph nodes.
    n_segments : int
        Segments per window.
    segment_len : int
        Samples per segment.
    d_model : int
        Width of a node representation after message passing.
    d_graph : int
        Width of the space the graphs are scored in.
    n_classes : int
        Number of sleep stages.
    coordinates : tuple of tuple of float
        Montage position ``(x, y, z)`` of each electrode, in channel order.
        Only fixed-geometry graphs (:class:`DistanceKernel`) read it.
    """

    n_channels: int
    n_segments: int
    segment_len: int
    d_model: int
    d_graph: int
    n_classes: int
    coordinates: tuple[tuple[float, float, float], ...]


def register(slot: str, name: str):
    """Class decorator that makes a component selectable by name from the config.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    name : str
        Name used as ``model.<slot>.name`` in the YAML files.

    Returns
    -------
    callable
        The decorator, which returns the class unchanged.

    Raises
    ------
    ValueError
        If the slot is unknown or the name is already taken.
    """
    if slot not in REGISTRY:
        raise ValueError(f"unknown slot {slot!r}; slots are {SLOTS}")

    def decorate(cls: type) -> type:
        """Add a class to the registry under ``slot`` and ``name``.

        Parameters
        ----------
        cls : type
            Component class.

        Returns
        -------
        type
            ``cls`` unchanged.
        """
        if name in REGISTRY[slot]:
            raise ValueError(f"{slot} component {name!r} is registered twice")
        REGISTRY[slot][name] = cls
        return cls

    return decorate


CONTEXT = {"dims", "context_dim", "in_dim"}


def check_spec(slot: str, spec: dict) -> type:
    """Check that a component specification names a registered component and fits it.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    spec : dict
        ``{"name": ..., **parameters}`` from ``model.<slot>`` in the config.

    Returns
    -------
    type
        The component class.

    Raises
    ------
    KeyError
        If the name is not registered.
    TypeError
        If parameters are unknown or missing.
    """
    parameters = {key: value for key, value in spec.items() if key != "name"}
    name = spec.get("name")
    if name not in REGISTRY[slot]:
        raise KeyError(f"unknown {slot} component {name!r}; registered: {sorted(REGISTRY[slot])}")
    cls = REGISTRY[slot][name]
    signature = inspect.signature(cls.__init__).parameters
    accepted = set(signature) - {"self"} - CONTEXT
    required = {key for key in accepted if signature[key].default is inspect.Parameter.empty}
    unknown, missing = set(parameters) - accepted, required - set(parameters)
    if unknown or missing:
        raise TypeError(f"model.{slot} ({name}): unknown {sorted(unknown)}, missing "
                        f"{sorted(missing)}; {name} takes {sorted(accepted)}")
    return cls


def check_model(model: dict) -> None:
    """Check every component specification of a model section.

    Parameters
    ----------
    model : dict
        Component specifications keyed by slot.
    """
    for slot in SLOTS:
        check_spec(slot, model[slot])


def build_component(slot: str, spec: dict, dims: Dimensions, **context) -> nn.Module:
    """Instantiate the component a config specification names.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    spec : dict
        ``{"name": ..., **parameters}`` from ``model.<slot>`` in the config.
    dims : Dimensions
        Shared sizes.
    **context
        Values produced by earlier components, such as the sequence width.

    Returns
    -------
    nn.Module
        The component.

    Raises
    ------
    KeyError
        If the name is not registered.
    TypeError
        If parameters are unknown or missing.
    """
    cls = check_spec(slot, spec)
    parameters = {key: value for key, value in spec.items() if key != "name"}
    accepted = set(inspect.signature(cls.__init__).parameters)
    return cls(dims, **parameters, **{k: v for k, v in context.items() if k in accepted})


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    length : int
        Number of positions.
    width : int
        Encoding width; must be even.

    Returns
    -------
    torch.Tensor
        A ``[length, width]`` table with values in ``[-1, 1]``.
    """
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = -np.log(10000.0) / width
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * scale)
    table = torch.zeros(length, width)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


@register("graph_learner", "residual_change")
class ResidualChange(nn.Module):
    """Stable graph ``L^S`` plus the change of what ``L^S`` cannot explain (Exp 2+).

    ``L^S`` comes from learned electrode embeddings and is shared by every
    window. ``Delta_t`` is the connectivity of the residual left after the
    stable graph propagates the segment, scaled by the share of energy it
    leaves unexplained, and optionally carried forward between segments.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    memory : bool
        Carry the change forward: ``Delta_t = (1 - a) Delta_{t-1} + a delta_t``
        with a learned ``a``. When false, every segment's change is independent.
    """

    def __init__(self, dims: Dimensions, memory: bool):
        super().__init__()
        self.embed_from = nn.Parameter(torch.randn(dims.n_channels, dims.d_graph))
        self.embed_to = nn.Parameter(torch.randn(dims.n_channels, dims.d_graph))
        self.project = nn.Linear(dims.segment_len, dims.d_graph, bias=False)
        self.use_memory = memory
        if memory:
            self.memory = nn.Parameter(torch.zeros(()))

    def stable_graph(self) -> torch.Tensor:
        """The stable graph, shared by every window.

        Returns
        -------
        torch.Tensor
            Symmetric scores ``[C, C]`` that depend only on parameters.
        """
        scores = self.embed_from @ self.embed_to.T / self.embed_from.shape[1] ** 0.5
        return 0.5 * (scores + scores.T)

    def change(self, cells: torch.Tensor, stable: torch.Tensor) -> torch.Tensor:
        """How each segment departs from what the stable graph predicts.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.
        stable : torch.Tensor
            Stable graph scores ``[C, C]``.

        Returns
        -------
        torch.Tensor
            Score changes ``[B, T, C, C]`` in ``[-1, 1]``.
        """
        residual = cells - torch.softmax(stable, dim=-1) @ cells
        projected = F.normalize(self.project(residual), dim=-1)
        scores = projected @ projected.mT
        left = (residual.norm(dim=-1) / cells.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0)
        return 0.5 * (scores + scores.mT) * (left.unsqueeze(-1) * left.unsqueeze(-2)).sqrt()

    def accumulate(self, delta: torch.Tensor) -> torch.Tensor:
        """Carry the change forward so a segment inherits the earlier ones.

        Parameters
        ----------
        delta : torch.Tensor
            Per-segment change ``[B, T, C, C]``.

        Returns
        -------
        torch.Tensor
            Accumulated change of the same shape; segment ``t`` depends only
            on segments ``1..t``.
        """
        alpha = torch.sigmoid(self.memory)
        kept = [delta[:, 0]]
        for step in range(1, delta.shape[1]):
            kept.append((1.0 - alpha) * kept[-1] + alpha * delta[:, step])
        return torch.stack(kept, dim=1)

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the graph of every segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.

        Returns
        -------
        scores : torch.Tensor
            Signed graph ``L^S + Delta_t``, ``[B, T, C, C]``.
        delta : torch.Tensor
            The change ``[B, T, C, C]``.
        stable : torch.Tensor
            The stable graph ``[C, C]``.
        """
        stable = self.stable_graph()
        delta = self.change(cells, stable)
        if self.use_memory:
            delta = self.accumulate(delta)
        return stable + delta, delta, stable


@register("graph_learner", "input_change")
class InputChange(ResidualChange):
    """Stable graph ``L^S`` plus a change computed from the segment itself, not from its residual.

    The ablation of the residual in ``residual_change``: the per-segment change is the cosine
    similarity of the projected input segments ``X_t[c]`` instead of the projected residuals
    ``X_t[c] - softmax(L^S) X_t``, so it no longer depends on what ``L^S`` already explains. The
    energy weight of ``residual_change`` (the share of an electrode's energy the stable graph
    leaves unexplained) has no counterpart without a residual and is fixed to one. Parameters,
    their names, the accumulation and ``A_t = L^S + Delta_t`` are those of ``residual_change``,
    so the two differ only in what the change is computed from.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    memory : bool
        Carry the change forward, as in ``residual_change``.
    """

    def change(self, cells: torch.Tensor, stable: torch.Tensor) -> torch.Tensor:
        """Signed similarity of the projected input segments of every electrode pair.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.
        stable : torch.Tensor
            Stable graph scores ``[C, C]``; unused, kept for the interface of ``residual_change``.

        Returns
        -------
        torch.Tensor
            Score changes ``[B, T, C, C]`` in ``[-1, 1]``.
        """
        projected = F.normalize(self.project(cells), dim=-1)
        scores = projected @ projected.mT
        return 0.5 * (scores + scores.mT)


@register("graph_learner", "stable_only")
class StableOnly(nn.Module):
    """The stable graph ``L^S`` alone, with no change term (ablation ``stable_only``).

    Every segment gets ``A_t = L^S``, built as in :class:`ResidualChange`
    from learned electrode embeddings, so ``Delta_t = 0``. Tests whether the
    change term adds anything (Q1).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    """

    def __init__(self, dims: Dimensions):
        super().__init__()
        self.embed_from = nn.Parameter(torch.randn(dims.n_channels, dims.d_graph))
        self.embed_to = nn.Parameter(torch.randn(dims.n_channels, dims.d_graph))

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repeat the stable graph for every segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``; only their shape is used.

        Returns
        -------
        scores : torch.Tensor
            ``L^S`` for every segment, ``[B, T, C, C]``.
        delta : torch.Tensor
            Zeros ``[B, T, C, C]``.
        stable : torch.Tensor
            ``L^S``, ``[C, C]``.
        """
        scores = self.embed_from @ self.embed_to.T / self.embed_from.shape[1] ** 0.5
        stable = 0.5 * (scores + scores.T)
        batch, steps = cells.shape[:2]
        delta = cells.new_zeros(batch, steps, *stable.shape)
        return stable.expand(batch, steps, -1, -1), delta, stable


@register("graph_learner", "distance_kernel")
class DistanceKernel(nn.Module):
    """A fixed graph from scalp distance, with nothing learned (ablation
    ``distance_graph``).

    The thresholded Gaussian kernel of Tang et al. (ICLR 2022,
    ``eeg-gnn-ssl/data/electrode_graph/generate_adj_mx.ipynb``): the
    electrodes are centred and scaled to a mean radius of one, so distances
    are in head radii as in their 10-20 table (Fp1-Fp2 = 0.618);
    ``A_ij = exp(-(d_ij / sigma)^2)`` with ``sigma`` the standard deviation of
    all pairwise distances, the diagonal included as in their code, and
    ``A_ij = 0`` where ``d_ij > threshold``. The graph is a buffer, the same
    for every segment, and has no sign: the signed convolution sees only
    positive edges.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes; ``coordinates`` gives the electrode positions.
    threshold : float
        Largest distance, in head radii, that keeps an edge.
    """

    def __init__(self, dims: Dimensions, threshold: float):
        super().__init__()
        coordinates = torch.tensor(dims.coordinates, dtype=torch.float64)
        coordinates = coordinates - coordinates.mean(0)
        coordinates = coordinates / coordinates.norm(dim=1).mean()
        distance = torch.cdist(coordinates, coordinates)
        graph = torch.exp(-(distance / distance.std(unbiased=False)) ** 2)
        graph = graph.masked_fill(distance > threshold, 0.0)
        self.register_buffer("graph", graph.float())

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repeat the distance graph for every segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``; only their shape is used.

        Returns
        -------
        scores : torch.Tensor
            The distance graph for every segment, ``[B, T, C, C]``.
        delta : torch.Tensor
            Zeros ``[B, T, C, C]``.
        stable : torch.Tensor
            The distance graph, ``[C, C]``.
        """
        batch, steps = cells.shape[:2]
        delta = cells.new_zeros(batch, steps, *self.graph.shape)
        return self.graph.expand(batch, steps, -1, -1), delta, self.graph


@register("graph_learner", "epoch_change")
class EpochChange(ResidualChange):
    """``L^S`` plus one change per 30 s window instead of one per segment (ablation
    ``epoch_graph``).

    The change of each segment is computed as in :class:`ResidualChange`
    (without memory), then averaged over the window's segments, and the
    same graph ``L^S + mean_t delta_t`` is given to every segment. This is
    the granularity of the published sleep graph models, one electrode graph
    per epoch. Every segment therefore sees the later segments of its
    window: this breaks the causal order inside the window on purpose, the
    only way to build one graph per epoch from the epoch. Parameter names
    are those of :class:`ResidualChange` without ``memory``.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    """

    def __init__(self, dims: Dimensions):
        super().__init__(dims, memory=False)

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one graph per window and repeat it for every segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.

        Returns
        -------
        scores : torch.Tensor
            ``L^S + mean_t delta_t`` for every segment, ``[B, T, C, C]``.
        delta : torch.Tensor
            The window's change, repeated over segments, ``[B, T, C, C]``.
        stable : torch.Tensor
            The stable graph ``[C, C]``.
        """
        stable = self.stable_graph()
        delta = self.change(cells, stable).mean(dim=1, keepdim=True)
        delta = delta.expand(-1, cells.shape[1], -1, -1)
        return stable + delta, delta, stable


@register("graph_learner", "no_graph")
class NoGraph(nn.Module):
    """No electrode graph: every edge is zero (ablations ``no_electrode_graph``,
    ``no_graph``).

    The signed convolution then has no neighbours of either sign, and each
    electrode keeps only its own-state path, so layer 1 becomes a
    per-electrode network. Lower bound for the electrode graph.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    """

    def __init__(self, dims: Dimensions):
        super().__init__()
        self.n_channels = dims.n_channels

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return an empty graph for every segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``; only their shape is used.

        Returns
        -------
        scores : torch.Tensor
            Zeros ``[B, T, C, C]``.
        delta : torch.Tensor
            Zeros ``[B, T, C, C]``.
        stable : torch.Tensor
            Zeros ``[C, C]``.
        """
        batch, steps = cells.shape[:2]
        empty = cells.new_zeros(batch, steps, self.n_channels, self.n_channels)
        return empty, empty, empty[0, 0]


@register("graph_conv", "vanilla")
class VanillaGraphConv(nn.Module):
    """Two GCN layers on ``softmax(L)``, symmetrised (Exp 1-4).

    The softmax makes every edge positive, so a strongly negative relationship
    reaches the convolution as a weak positive edge.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    dropout : float
        Dropout after each layer.
    """

    def __init__(self, dims: Dimensions, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(dims.segment_len, dims.d_model)
        self.layer2 = nn.Linear(dims.d_model, dims.d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, cells: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Propagate node features over the graph of each segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.
        scores : torch.Tensor
            Signed graph scores ``[B, T, C, C]``.

        Returns
        -------
        torch.Tensor
            Node representations ``[B, T, C, d_model]``.
        """
        adjacency = torch.softmax(scores, dim=-1)
        adjacency = 0.5 * (adjacency + adjacency.mT)
        hidden = self.drop(F.relu(self.layer1(adjacency @ cells)))
        return self.drop(F.relu(self.layer2(adjacency @ hidden)))


class SignedGraphConvLayer(nn.Module):
    """One signed graph convolution layer (Derr, Ma & Tang, ICDM 2018, Eq. 3-6).

    Every electrode keeps a balanced and an unbalanced state. The balanced
    state gathers from positive neighbours and from the unbalanced state of
    negative neighbours, and the unbalanced state the other way round, so a
    negative edge means "opposes".

    Parameters
    ----------
    in_dim : int
        Width of the incoming features, per state.
    out_dim : int
        Width of each outgoing state.
    first : bool
        Layer 1 reads the raw node features (``2 * in_dim`` inputs) instead of
        two states (``3 * in_dim`` inputs).
    """

    def __init__(self, in_dim: int, out_dim: int, first: bool):
        super().__init__()
        self.first = first
        width = (2 if first else 3) * in_dim
        self.balanced = nn.Linear(width, out_dim)
        self.unbalanced = nn.Linear(width, out_dim)

    def forward(self, positive: torch.Tensor, negative: torch.Tensor, balanced: torch.Tensor,
                unbalanced: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate over positive and negative neighbours separately.

        Parameters
        ----------
        positive, negative : torch.Tensor
            Row-normalised adjacencies ``[B, T, C, C]``.
        balanced : torch.Tensor
            Balanced state ``[B, T, C, in_dim]``, or the raw node features in layer 1.
        unbalanced : torch.Tensor, optional
            Unbalanced state ``[B, T, C, in_dim]``; unused in layer 1.

        Returns
        -------
        tuple of torch.Tensor
            New balanced and unbalanced states, each ``[B, T, C, out_dim]``.
        """
        if self.first:
            to_balanced = torch.cat([positive @ balanced, balanced], -1)
            to_unbalanced = torch.cat([negative @ balanced, balanced], -1)
        else:
            to_balanced = torch.cat([positive @ balanced, negative @ unbalanced, balanced], -1)
            to_unbalanced = torch.cat([positive @ unbalanced, negative @ balanced, unbalanced], -1)
        return torch.tanh(self.balanced(to_balanced)), torch.tanh(self.unbalanced(to_unbalanced))


@register("graph_conv", "signed")
class SignedGraphConv(nn.Module):
    """Two signed graph convolution layers that keep the sign of ``L`` (Exp 5+).

    Each state is ``d_model / 2`` wide, so the output keeps the width of the
    vanilla convolution.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    dropout : float
        Dropout after each layer.
    """

    def __init__(self, dims: Dimensions, dropout: float):
        super().__init__()
        half = dims.d_model // 2
        self.layer1 = SignedGraphConvLayer(dims.segment_len, half, first=True)
        self.layer2 = SignedGraphConvLayer(half, half, first=False)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def signed_adjacency(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a signed graph into row-normalised positive and negative halves.

        The diagonal is dropped because each layer appends a node's own state.

        Parameters
        ----------
        scores : torch.Tensor
            Signed scores ``[..., C, C]``.

        Returns
        -------
        tuple of torch.Tensor
            Positive and negative adjacencies, non-negative, rows summing to
            one, or to zero for a node without neighbours of that sign.
        """
        eye = torch.eye(scores.shape[-1], dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(eye, 0.0)
        positive, negative = scores.clamp(min=0), (-scores).clamp(min=0)
        return (positive / positive.sum(-1, keepdim=True).clamp(min=1e-6),
                negative / negative.sum(-1, keepdim=True).clamp(min=1e-6))

    def forward(self, cells: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Propagate node features over the signed graph of each segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.
        scores : torch.Tensor
            Signed graph scores ``[B, T, C, C]``.

        Returns
        -------
        torch.Tensor
            ``[balanced | unbalanced]`` node representations ``[B, T, C, d_model]``.
        """
        positive, negative = self.signed_adjacency(scores)
        balanced, unbalanced = self.layer1(positive, negative, cells)
        balanced, unbalanced = self.layer2(positive, negative,
                                           self.drop(balanced), self.drop(unbalanced))
        return self.drop(torch.cat([balanced, unbalanced], -1))


@register("graph_conv", "unsigned")
class UnsignedGraphConv(SignedGraphConv):
    """The signed convolution with every edge treated as positive (sign ablation).

    The graph learner still produces the signed ``A_t``, but the convolution
    sees only its magnitudes: the positive adjacency is the row-normalised
    ``|A_t|`` and the negative adjacency is empty, so a strongly negative edge
    pulls in its neighbour like a strongly positive one and no neighbour reaches
    a node through the "opposes" path. The layers, their parameter names and
    the parameter count are those of ``signed``, so a run with this component
    differs from Exp 15 only in whether the convolution can use edge signs.
    Training it answers what the test-time sign edits cannot: whether a model
    trained without signs is worse, rather than whether a model trained with
    them relies on them.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    dropout : float
        Dropout after each layer.
    """

    @staticmethod
    def signed_adjacency(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-normalised edge magnitudes and an empty negative adjacency.

        Parameters
        ----------
        scores : torch.Tensor
            Signed scores ``[..., C, C]``.

        Returns
        -------
        tuple of torch.Tensor
            ``|scores|`` without the diagonal, rows summing to one, and zeros of
            the same shape.
        """
        eye = torch.eye(scores.shape[-1], dtype=torch.bool, device=scores.device)
        magnitude = scores.masked_fill(eye, 0.0).abs()
        return (magnitude / magnitude.sum(-1, keepdim=True).clamp(min=1e-6),
                torch.zeros_like(magnitude))


def signed_magnitudes(scores: torch.Tensor,
                      allowed: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a signed graph into the magnitudes of its positive and negative edges.

    The diagonal is always dropped, because the signed layers append a node's
    own state separately.

    Parameters
    ----------
    scores : torch.Tensor
        Signed scores ``[..., N, N]``; row ``i`` holds the edges node ``i`` receives.
    allowed : torch.Tensor, optional
        Boolean ``[N, N]``, true where an edge may exist, such as a causal mask.

    Returns
    -------
    tuple of torch.Tensor
        Positive and negative magnitudes, both non-negative, zero where there is
        no edge of that sign.
    """
    keep = ~torch.eye(scores.shape[-1], dtype=torch.bool, device=scores.device)
    if allowed is not None:
        keep = keep & allowed
    scores = scores.masked_fill(~keep, 0.0)
    return scores.clamp(min=0), (-scores).clamp(min=0)


def masked_softmax(logits: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis restricted to existing edges.

    A row without any edge gets all-zero weights instead of NaN, so a node
    with no neighbour of one sign receives nothing from that sign, as in the
    row-normalised signed convolution.

    Parameters
    ----------
    logits : torch.Tensor
        Attention logits ``[..., N, N]``.
    edges : torch.Tensor
        Boolean, broadcastable to ``logits``; true where an edge exists.

    Returns
    -------
    torch.Tensor
        Weights of the same shape; each row sums to one, or to zero if empty.
    """
    filled = logits.masked_fill(~edges, float("-inf"))
    peak = filled.amax(-1, keepdim=True).detach()
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(filled - peak)
    return weights / weights.sum(-1, keepdim=True).clamp(min=1e-6)


class SignedAttention(nn.Module):
    """Attention over positive and over negative neighbours, on top of the edge weights.

    For each sign ``s`` and head ``k`` the logit of edge ``j -> i`` is the
    additive attention of GAT (Velickovic et al., 2018), as used by SiGAT
    (Huang et al., 2019) and SNEA (Li et al., 2020), plus the log of the
    learned edge magnitude::

        e_ij = LeakyReLU(a_target . W x_i + a_source . W x_j) + log |L_ij|

    and the weights are a softmax over the neighbours of that sign. The
    attention vectors start at zero, so at initialisation the weights are
    ``|L_ij| / sum_j |L_ij|``, exactly the row-normalised adjacency of the
    signed convolution; attention then learns to re-weight neighbours by their
    current features. Keeping ``log |L_ij|`` matters: without it the graph
    would enter only through its sign pattern, which has no gradient, and the
    graph learner would stop learning from the task. The heads are averaged.

    Parameters
    ----------
    in_dim : int
        Width of the node features the attention reads.
    attention_dim : int
        Width of the attention space of each head.
    heads : int
        Attention heads per sign.
    negative_slope : float
        Slope of the LeakyReLU for negative inputs.
    """

    def __init__(self, in_dim: int, attention_dim: int, heads: int, negative_slope: float):
        super().__init__()
        self.heads, self.attention_dim = heads, attention_dim
        self.negative_slope = negative_slope
        self.project = nn.Linear(in_dim, 2 * heads * attention_dim, bias=False)
        self.target = nn.Parameter(torch.zeros(2, heads, attention_dim))
        self.source = nn.Parameter(torch.zeros(2, heads, attention_dim))

    def forward(self, features: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention weights over the positive and over the negative neighbours.

        Parameters
        ----------
        features : torch.Tensor
            Node features ``[..., N, in_dim]``.
        positive, negative : torch.Tensor
            Edge magnitudes ``[..., N, N]`` from :func:`signed_magnitudes`.

        Returns
        -------
        tuple of torch.Tensor
            Positive and negative weights ``[..., N, N]``, rows summing to one,
            or to zero for a node without neighbours of that sign.
        """
        projected = self.project(features).unflatten(-1, (2, self.heads, self.attention_dim))
        target = (projected * self.target).sum(-1).movedim(-3, -1)
        source = (projected * self.source).sum(-1).movedim(-3, -1)
        logits = F.leaky_relu(target.unsqueeze(-1) + source.unsqueeze(-2), self.negative_slope)
        magnitude = torch.stack([positive, negative], -3).unsqueeze(-3)
        logits = logits + torch.log(magnitude.clamp(min=1e-12))
        weights = masked_softmax(logits, magnitude > 0).mean(-3)
        return weights[..., 0, :, :], weights[..., 1, :, :]


@register("graph_conv", "signed_attention")
class SignedGraphAttention(nn.Module):
    """Two signed graph attention layers on ``L^S + Delta_t`` (Exp 7+).

    The same balanced / unbalanced update as the signed convolution (Exp 5,
    Derr, Ma & Tang 2018), but each electrode weighs its positive and its
    negative neighbours with :class:`SignedAttention` instead of with the fixed
    row-normalised edge weights. This is the balance-theory form of a signed
    graph attention network (SNEA, Li et al., AAAI 2020); SiGAT's triangle
    motifs are not used, because on a dense 83-node graph almost every pair
    shares every third electrode. At initialisation the layer computes exactly
    what the signed convolution computes.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    dropout : float
        Dropout after each layer.
    heads : int
        Attention heads per sign and layer.
    attention_dim : int
        Width of the attention space of each head.
    negative_slope : float
        Slope of the attention LeakyReLU.
    """

    def __init__(self, dims: Dimensions, dropout: float, heads: int, attention_dim: int,
                 negative_slope: float):
        super().__init__()
        half = dims.d_model // 2
        self.layer1 = SignedGraphConvLayer(dims.segment_len, half, first=True)
        self.layer2 = SignedGraphConvLayer(half, half, first=False)
        self.drop = nn.Dropout(dropout)
        self.attend1 = SignedAttention(dims.segment_len, attention_dim, heads, negative_slope)
        self.attend2 = SignedAttention(2 * half, attention_dim, heads, negative_slope)

    def forward(self, cells: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Propagate node features over the signed graph of each segment.

        Parameters
        ----------
        cells : torch.Tensor
            Node features ``[B, T, C, L]``.
        scores : torch.Tensor
            Signed graph scores ``[B, T, C, C]``.

        Returns
        -------
        torch.Tensor
            ``[balanced | unbalanced]`` node representations ``[B, T, C, d_model]``.
        """
        positive, negative = signed_magnitudes(scores, None)
        balanced, unbalanced = self.layer1(*self.attend1(cells, positive, negative), cells)
        balanced, unbalanced = self.drop(balanced), self.drop(unbalanced)
        states = torch.cat([balanced, unbalanced], -1)
        balanced, unbalanced = self.layer2(*self.attend2(states, positive, negative),
                                           balanced, unbalanced)
        return self.drop(torch.cat([balanced, unbalanced], -1))


@register("position", "none")
class NoPosition(nn.Module):
    """Leave the node representations unchanged.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes (unused).
    """

    def __init__(self, dims: Dimensions):
        super().__init__()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the input.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.

        Returns
        -------
        torch.Tensor
            ``hidden`` itself.
        """
        return hidden


@register("position", "segment")
class SegmentPosition(nn.Module):
    """Add a sinusoidal encoding of the segment index to every node (Exp 6).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    """

    def __init__(self, dims: Dimensions):
        super().__init__()
        self.register_buffer("table", sinusoidal_positions(dims.n_segments, dims.d_model),
                             persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Add the encoding of segment ``t`` to every electrode of segment ``t``.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.

        Returns
        -------
        torch.Tensor
            Same shape, with the segment position added.
        """
        return hidden + self.table.unsqueeze(1)


class RecurrentSequence(nn.Module):
    """Read the segment representations in order with a recurrent network.

    Each segment's node representations are flattened into one vector. The
    network runs in one direction only, so step ``t`` sees segments ``1..t``.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    cell : type
        ``nn.GRU`` or ``nn.LSTM``.
    hidden : int
        Hidden width.
    layers : int
        Stacked layers.
    dropout : float
        Dropout between stacked layers.
    """

    def __init__(self, dims: Dimensions, cell: type, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = cell(dims.n_channels * dims.d_model, hidden, num_layers=layers,
                        batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.output_dim = hidden

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run over the segments.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``.
        """
        output, _ = self.rnn(hidden.flatten(2))
        return output


@register("sequence", "gru")
class GRUSequence(RecurrentSequence):
    """GRU over the segments (Exp 4+).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Hidden width.
    layers : int
        Stacked layers.
    dropout : float
        Dropout between stacked layers.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, dropout: float):
        super().__init__(dims, nn.GRU, hidden, layers, dropout)


@register("sequence", "lstm")
class LSTMSequence(RecurrentSequence):
    """LSTM over the segments (Exp 1-3).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Hidden width.
    layers : int
        Stacked layers.
    dropout : float
        Dropout between stacked layers.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, dropout: float):
        super().__init__(dims, nn.LSTM, hidden, layers, dropout)


class SignedEdgeLearning(nn.Module):
    """Learned signed edges between a segment and every earlier one (Exp 8+).

    The edge learning shared by the layer-2 components; a subclass creates
    ``embed``, ``sender``, ``receiver``, ``edge_out``, ``temperature``, the
    ``causal`` buffer (true where ``s < t``), ``random_edges`` and ``penalty``.
    See :class:`SegmentGraph` for the method and the reasons behind it.
    """

    EDGE_TYPES = ("positive", "negative")

    def edge_logits(self, nodes: torch.Tensor) -> torch.Tensor:
        """Score every (receiver, sender) pair of segments.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment nodes ``[B, T, hidden]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, T, T, 2]`` of the edge types positive / negative;
            entry ``[b, t, s]`` is the edge from ``s`` to ``t``.
        """
        pair = self.sender(nodes).unsqueeze(1) + self.receiver(nodes).unsqueeze(2)
        return self.edge_out(F.relu(pair))

    def edge_types(self, nodes: torch.Tensor) -> torch.Tensor:
        """One edge type per pair of segments, see :meth:`sample`.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment nodes ``[B, T, hidden]``.

        Returns
        -------
        torch.Tensor
            One-hot edge types ``[B, T, T, 2]``.
        """
        return self.sample(self.edge_logits(nodes))

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """One edge type per pair: sampled in training, most likely in evaluation.

        When ``random_edges`` holds one probability per type, the types are drawn
        from them instead, independently of the window; this is the
        random-graph control of the evaluation.

        Parameters
        ----------
        logits : torch.Tensor
            Edge-type logits ``[B, T, T, 2]`` from :meth:`edge_logits`.

        Returns
        -------
        torch.Tensor
            One-hot edge types ``[B, T, T, 2]``; in training a straight-through
            Gumbel-softmax sample, whose gradient is that of the soft sample.
        """
        if self.random_edges is not None:
            drawn = torch.multinomial(self.random_edges.to(logits.device),
                                      logits[..., 0].numel(), replacement=True)
            return F.one_hot(drawn.view(logits.shape[:-1]), len(self.EDGE_TYPES)).to(logits)
        if self.training:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return F.one_hot(logits.argmax(-1), len(self.EDGE_TYPES)).to(logits)

    def kl_to_uniform(self, logits: torch.Tensor) -> torch.Tensor:
        """KL divergence of the edge-type distribution from the uniform one (NRI).

        Parameters
        ----------
        logits : torch.Tensor
            Edge-type logits ``[B, T, T, 2]``.

        Returns
        -------
        torch.Tensor
            Scalar mean over the windows and the earlier -> later pairs, between
            0 (both types equally likely) and ``log 2`` (one type certain).
        """
        log_probability = logits[:, self.causal].log_softmax(-1)
        entropy_gap = (log_probability.exp() * log_probability).sum(-1)
        return entropy_gap.mean() + np.log(len(self.EDGE_TYPES))


@register("sequence", "segment_graph")
class SegmentGraph(SignedEdgeLearning):
    """Layer 2: a learned, causal, signed graph whose nodes are the segments (Exp 8).

    Layer 1 (``graph_conv``) gives every segment a representation of its
    electrode graph. Here each segment becomes one node, embedded from its
    flattened electrode representations as the GRU input was, and the edges
    between segments are learned for every window. There are no fixed edges.

    Edge learning follows the link predictor of GTS (Shang, Chen & Bi, ICLR
    2021) and TimeGNN (Xu, Kosma & Vazirgiannis, 2023), with the edge types of
    NRI (Kipf et al., ICML 2018). For every pair of a receiving segment ``t``
    and an earlier segment ``s``, an MLP of the concatenated pair gives the
    logits of two edge types, positive and negative. There is no "none" type:
    with one, the learner switched every edge off within three epochs, because
    the zero message is the easiest way to silence early noisy neighbours, and
    NRI and GTS need a prior on the edge distribution to prevent that. Every
    earlier segment is therefore connected, with a learned sign, and the
    attention decides how much each one counts. While training,
    one type is sampled with the straight-through Gumbel-softmax
    (``F.gumbel_softmax(hard=True)``, the function both official codes use), so
    the forward pass sees a discrete graph and the gradient reaches the edge
    learner. In evaluation the most likely type is taken, so predictions are
    deterministic. The first MLP layer is applied as a sender part plus a
    receiver part, which equals one linear layer on the concatenation.

    Only pairs with ``s < t`` are scored, so segment ``t`` receives from
    segments ``1..t-1`` and keeps its own state; later segments cannot reach it.

    Two signed graph attention layers propagate over the learned graph. Each
    attends over all earlier segments, separately for the positive and the
    negative side (:class:`SignedAttention`), and the attention weight of an
    edge is multiplied by its sampled ``{0, 1}`` indicator of that sign, so
    an edge passes its message on one side only. The indicator of the other
    side still has a gradient, because it multiplies a non-zero message, as in
    GTS. The weights are not renormalised after gating, which would divide by
    zero for a segment without edges of one sign.

    Without a counterweight the edge learner saturates within one epoch to a
    single type for every pair (p(+) about 0.001 everywhere), so the graph
    stops depending on the window. Like NRI, whose loss always contains the KL
    divergence of the edge distribution from a uniform prior, every forward
    pass leaves that KL, averaged over the pairs, in ``penalty``; the training
    loop adds it with the weight ``graph_penalty``.

    The output of segment ``t`` takes the place of the GRU output at step
    ``t``: the head reads the last segment and the decoder reads every one.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Width of a segment node; split into balanced and unbalanced halves.
    edge_hidden : int
        Width of the hidden layer of the edge MLP.
    temperature : float
        Gumbel-softmax temperature.
    heads : int
        Attention heads per sign and layer.
    attention_dim : int
        Width of the attention space of each head.
    negative_slope : float
        Slope of the attention LeakyReLU.
    dropout : float
        Dropout after each layer.
    """

    def __init__(self, dims: Dimensions, hidden: int, edge_hidden: int, temperature: float,
                 heads: int, attention_dim: int, negative_slope: float, dropout: float):
        super().__init__()
        half = hidden // 2
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.sender = nn.Linear(hidden, edge_hidden)
        self.receiver = nn.Linear(hidden, edge_hidden, bias=False)
        self.edge_out = nn.Linear(edge_hidden, len(self.EDGE_TYPES))
        self.temperature = temperature
        self.layer1 = SignedGraphConvLayer(hidden, half, first=True)
        self.layer2 = SignedGraphConvLayer(half, half, first=False)
        self.attend1 = SignedAttention(hidden, attention_dim, heads, negative_slope)
        self.attend2 = SignedAttention(2 * half, attention_dim, heads, negative_slope)
        self.drop = nn.Dropout(dropout)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.output_dim = 2 * half
        self.random_edges: torch.Tensor | None = None
        self.penalty: torch.Tensor | None = None

    def importance(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """How much each earlier segment counts for each segment in the first layer.

        The same gated attention the first layer passes messages with, exposed
        for the figures; it does not change :meth:`forward`.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment nodes ``[B, T, hidden]``.

        Returns
        -------
        tuple of torch.Tensor
            Importance of the positive and of the negative edges ``[B, T, T]``.
        """
        causal = self.causal.to(nodes.dtype)
        edges = self.edge_types(nodes) * causal.unsqueeze(-1)
        positive, negative = self.attend1(nodes, causal, causal)
        return positive * edges[..., 0], negative * edges[..., 1]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Learn the segment graph of every window and propagate over it.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One ``[balanced | unbalanced]`` output per segment, ``[B, T, output_dim]``.
        """
        nodes = self.embed(hidden.flatten(2))
        causal = self.causal.to(nodes.dtype)
        logits = self.edge_logits(nodes)
        self.penalty = self.kl_to_uniform(logits)
        edges = self.sample(logits) * causal.unsqueeze(-1)
        positive, negative = edges[..., 0], edges[..., 1]

        attention = self.attend1(nodes, causal, causal)
        balanced, unbalanced = self.layer1(attention[0] * positive, attention[1] * negative, nodes)
        balanced, unbalanced = self.drop(balanced), self.drop(unbalanced)
        states = torch.cat([balanced, unbalanced], -1)
        attention = self.attend2(states, causal, causal)
        balanced, unbalanced = self.layer2(attention[0] * positive, attention[1] * negative,
                                           balanced, unbalanced)
        return self.drop(torch.cat([balanced, unbalanced], -1))


class GraphBiasedBlock(nn.Module):
    """One pre-norm Transformer block whose attention is biased by a signed graph.

    The attention logit of head ``h`` from segment ``s`` to segment ``t`` is

        q_t . k_s / sqrt(d_head) + beta+_h * I+_ts + beta-_h * I-_ts

    where ``I+`` and ``I-`` are the importances of the positive and negative
    edges of the learned segment graph. Adding a learned function of the graph
    to the attention logits is how Graphormer (Ying et al., NeurIPS 2021)
    brings graph structure into a Transformer. ``beta`` starts at zero, so the
    block starts as a plain causal Transformer block and learns how much to
    follow the graph. Softmax weights stay non-negative: the sign of an edge
    decides whether it raises or lowers attention, not whether it carries an
    opposing message.

    Parameters
    ----------
    width : int
        Token width.
    heads : int
        Attention heads.
    ff_mult : int
        Width of the feed-forward hidden layer relative to ``width``.
    dropout : float
        Dropout of the attention weights and of both residual branches.
    """

    def __init__(self, width: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.graph_weight = nn.Parameter(torch.zeros(2, heads))
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_mult * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_mult * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                allowed: torch.Tensor) -> torch.Tensor:
        """Attend over the allowed segments with the graph bias, then the feed-forward layer.

        Parameters
        ----------
        tokens : torch.Tensor
            Segment tokens ``[B, T, width]``.
        positive, negative : torch.Tensor or None
            Edge importances ``[B, T, T]``; ``None`` removes the graph bias.
        allowed : torch.Tensor
            Boolean ``[T, T]``, true where segment ``t`` may attend to ``s``.

        Returns
        -------
        torch.Tensor
            Updated tokens ``[B, T, width]``.
        """
        query, key, value = (self.qkv(self.norm1(tokens))
                             .unflatten(-1, (3, self.heads, -1)).permute(2, 0, 3, 1, 4))
        logits = query @ key.mT / query.shape[-1] ** 0.5
        if positive is not None:
            logits = logits + (self.graph_weight[0].view(-1, 1, 1) * positive.unsqueeze(1)
                               + self.graph_weight[1].view(-1, 1, 1) * negative.unsqueeze(1))
        weights = self.drop(logits.masked_fill(~allowed, float("-inf")).softmax(-1))
        attended = (weights @ value).transpose(1, 2).flatten(2)
        tokens = tokens + self.drop(self.out(attended))
        return tokens + self.drop(self.ff(self.norm2(tokens)))


class ContinuousSignedEdges(nn.Module):
    """Continuous signed edges between a segment and every earlier one (Exp 9a+).

    Exp 8's discrete edge signs collapsed to one sign for every pair in every
    fold, and random signs scored the same. In the two-state signed layers,
    flipping every sign only swaps the balanced and unbalanced states, which
    the weights can absorb, so with straight-through Gumbel samples nothing
    holds different signs apart and the samples saturate. No graph neural
    network we found learns free discrete signs; FAGCN (Bo et al., AAAI 2021)
    learns a continuous signed coefficient per edge from the pair of node
    features, ``tanh(g . [h_i || h_j])`` in ``[-1, 1]``, and GGCN (Yan et al.,
    2021) takes the sign from a continuous similarity. Here the same link
    predictor as Exp 8 ends in one output, the scores of each segment are
    centred over its earlier segments, and a ``tanh`` bounds them:

        w_ts = tanh(score_ts - mean_{s' < t} score_ts'),   score = MLP([n_t || n_s])

    Without the centring, continuous weights collapsed as the discrete signs
    had: after three epochs 99.95% were negative with ``|w|`` near 1, because a
    single bias moves every pair to one side and larger gates pass more
    message. The centring removes exactly that global offset, which the task
    cannot pin down; each segment's earlier segments are rated against each
    other, positive meaning more related than the average earlier segment
    and negative less. The second segment, with one earlier segment, gets a
    zero weight. The gradient reaches every pair without sampling, so no KL
    penalty is needed. A subclass creates
    ``embed``, ``sender``, ``receiver``, ``edge_out`` (one output), the
    ``causal`` buffer (true where ``s < t``) and ``shuffle_edges``.
    """

    def edge_weights(self, nodes: torch.Tensor) -> torch.Tensor:
        """Signed weight of every edge from an earlier segment, centred per segment.

        When ``shuffle_edges`` is true, the weights of each window are randomly
        permuted over its earlier -> later pairs: the same values, placed at
        random. This is the random-graph control of the evaluation.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment nodes ``[B, T, hidden]``.

        Returns
        -------
        torch.Tensor
            ``[B, T, T]`` in ``[-1, 1]``; entry ``[b, t, s]`` is the edge from
            ``s`` to ``t``, zero where ``s >= t``.
        """
        pair = self.sender(nodes).unsqueeze(1) + self.receiver(nodes).unsqueeze(2)
        scores = self.edge_out(F.relu(pair)).squeeze(-1) * self.causal
        centre = scores.sum(-1, keepdim=True) / self.causal.sum(-1, keepdim=True).clamp(min=1)
        weights = torch.tanh(scores - centre) * self.causal
        if self.shuffle_edges:
            values = weights[:, self.causal]
            order = torch.rand_like(values).argsort(-1)
            weights = torch.zeros_like(weights)
            weights[:, self.causal] = values.gather(1, order)
        return weights


@register("sequence", "continuous_segment_graph")
class ContinuousSegmentGraph(ContinuousSignedEdges):
    """Exp 8's segment graph with continuous signed edges (Exp 9a).

    Everything is :class:`SegmentGraph` except the edges: instead of a sampled
    sign per pair, :class:`ContinuousSignedEdges` gives a weight in
    ``[-1, 1]`` that gates the attention of both signed layers. Segment ``t``
    still receives only from segments ``1..t-1`` and keeps its own state.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Width of a segment node; split into balanced and unbalanced halves.
    edge_hidden : int
        Width of the hidden layer of the edge MLP.
    heads : int
        Attention heads per sign and layer.
    attention_dim : int
        Width of the attention space of each head.
    negative_slope : float
        Slope of the attention LeakyReLU.
    dropout : float
        Dropout after each layer.
    """

    def __init__(self, dims: Dimensions, hidden: int, edge_hidden: int, heads: int,
                 attention_dim: int, negative_slope: float, dropout: float):
        super().__init__()
        half = hidden // 2
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.sender = nn.Linear(hidden, edge_hidden)
        self.receiver = nn.Linear(hidden, edge_hidden, bias=False)
        self.edge_out = nn.Linear(edge_hidden, 1)
        self.layer1 = SignedGraphConvLayer(hidden, half, first=True)
        self.layer2 = SignedGraphConvLayer(half, half, first=False)
        self.attend1 = SignedAttention(hidden, attention_dim, heads, negative_slope)
        self.attend2 = SignedAttention(2 * half, attention_dim, heads, negative_slope)
        self.drop = nn.Dropout(dropout)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.output_dim = 2 * half
        self.shuffle_edges = False

    def importance(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated attention of the first layer, for the figures.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment nodes ``[B, T, hidden]``.

        Returns
        -------
        tuple of torch.Tensor
            Importance of the positive and of the negative edges ``[B, T, T]``.
        """
        causal = self.causal.to(nodes.dtype)
        weights = self.edge_weights(nodes)
        positive, negative = self.attend1(nodes, causal, causal)
        return positive * weights.clamp(min=0), negative * (-weights).clamp(min=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Learn the continuous segment graph of every window and propagate over it.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One ``[balanced | unbalanced]`` output per segment, ``[B, T, output_dim]``.
        """
        nodes = self.embed(hidden.flatten(2))
        causal = self.causal.to(nodes.dtype)
        weights = self.edge_weights(nodes)
        positive, negative = weights.clamp(min=0), (-weights).clamp(min=0)

        attention = self.attend1(nodes, causal, causal)
        balanced, unbalanced = self.layer1(attention[0] * positive, attention[1] * negative, nodes)
        balanced, unbalanced = self.drop(balanced), self.drop(unbalanced)
        states = torch.cat([balanced, unbalanced], -1)
        attention = self.attend2(states, causal, causal)
        balanced, unbalanced = self.layer2(attention[0] * positive, attention[1] * negative,
                                           balanced, unbalanced)
        return self.drop(torch.cat([balanced, unbalanced], -1))


@register("sequence", "continuous_graph_transformer")
class ContinuousGraphTransformer(ContinuousSignedEdges):
    """Exp 9a's continuous segment graph guides a causal Transformer (Exp 9b).

    The segment tokens are the layer-1 features, embedded as in Exp 8. From the
    same tokens the continuous signed graph of Exp 9a is learned, and one
    signed attention over the earlier segments, gated by the edge weights,
    gives every pair an importance on the positive side ``I+`` and on the
    negative side ``I-``. A stack of :class:`GraphBiasedBlock` runs over the
    layer-1 tokens; each head adds ``beta+ I+ + beta- I-`` to its attention
    logits (Graphormer-style bias, ``beta`` starting at zero). Segment ``t``
    attends to segments ``1..t`` only; the head reads the last segment and the
    decoder reads every one. Setting ``graph_bias`` to false removes the graph
    from every block, for the evaluation.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Token width.
    layers : int
        Transformer blocks.
    heads : int
        Attention heads per block.
    ff_mult : int
        Feed-forward width relative to ``hidden``.
    dropout : float
        Dropout inside the blocks.
    edge_hidden : int
        Width of the hidden layer of the edge MLP.
    graph_heads : int
        Heads of the signed attention that measures edge importance.
    attention_dim : int
        Width of the attention space of each of those heads.
    negative_slope : float
        Slope of their LeakyReLU.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, heads: int, ff_mult: int,
                 dropout: float, edge_hidden: int, graph_heads: int, attention_dim: int,
                 negative_slope: float):
        super().__init__()
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.sender = nn.Linear(hidden, edge_hidden)
        self.receiver = nn.Linear(hidden, edge_hidden, bias=False)
        self.edge_out = nn.Linear(edge_hidden, 1)
        self.attend = SignedAttention(hidden, attention_dim, graph_heads, negative_slope)
        self.blocks = nn.ModuleList(GraphBiasedBlock(hidden, heads, ff_mult, dropout)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.register_buffer("allowed", steps.unsqueeze(1) >= steps, persistent=False)
        self.output_dim = hidden
        self.shuffle_edges = False
        self.graph_bias = True

    def importance(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed attention over the earlier segments, gated by the edge weights.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment tokens ``[B, T, hidden]``.

        Returns
        -------
        tuple of torch.Tensor
            Importance of the positive and of the negative edges ``[B, T, T]``.
        """
        causal = self.causal.to(nodes.dtype)
        weights = self.edge_weights(nodes)
        positive, negative = self.attend(nodes, causal, causal)
        return positive * weights.clamp(min=0), negative * (-weights).clamp(min=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the graph-guided causal Transformer over the segments.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``.
        """
        tokens = self.embed(hidden.flatten(2))
        positive, negative = self.importance(tokens)
        if not self.graph_bias:
            positive = negative = None
        for block in self.blocks:
            tokens = block(tokens, positive, negative, self.allowed)
        return self.norm(tokens)


@register("sequence", "stable_change_graph_transformer")
class StableChangeGraphTransformer(nn.Module):
    """Layer 1's stable-plus-change algorithm builds the segment graph (Exp 10).

    The segment graph is built the way :class:`ResidualChange` builds the
    electrode graph, with segments as nodes and only earlier segments as
    neighbours:

        G_ts = S^S_ts + Delta_ts,   s < t

    - ``S^S`` comes from learned embeddings of the segment positions and is
      shared by every window: how a segment position generally relates to an
      earlier one.
    - ``Delta`` is the change: the residual of each segment after ``S^S``
      predicts it from the earlier segments ``1..t-1`` (softmax over them),
      projected and compared by cosine similarity, and scaled by the share of
      each segment's energy the stable graph leaves unexplained. Unlike layer
      1, a segment is not predicted from itself: the first segment would then
      be predicted perfectly, its residual would be zero, and the square root
      of the energy share would have an infinite gradient. With nothing
      before it, the first segment keeps its whole signal as residual.

    The sign of an edge therefore comes from the data, as in layer 1 and in
    GGCN (Yan et al., 2021), not from a free parameter, which is what made the
    learned signs of Exp 8 collapse. As in layer 1, one signed attention
    (:class:`SignedAttention`, with ``log |G|`` in its logits) over the positive
    and over the negative earlier segments gives each pair an importance, and
    these bias the attention logits of the causal Transformer of Exp 9b, which
    runs on the layer-1 segment tokens. Setting ``graph_bias`` to false removes
    the graph from every block, and ``shuffle_edges`` places each window's
    edges at random, for the evaluation.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes; ``d_graph`` is the width of the graph space.
    hidden : int
        Token width.
    layers : int
        Transformer blocks.
    heads : int
        Attention heads per block.
    ff_mult : int
        Feed-forward width relative to ``hidden``.
    dropout : float
        Dropout inside the blocks.
    graph_heads : int
        Heads of the signed attention that measures edge importance.
    attention_dim : int
        Width of the attention space of each of those heads.
    negative_slope : float
        Slope of their LeakyReLU.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, heads: int, ff_mult: int,
                 dropout: float, graph_heads: int, attention_dim: int, negative_slope: float):
        super().__init__()
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.embed_from = nn.Parameter(torch.randn(dims.n_segments, dims.d_graph))
        self.embed_to = nn.Parameter(torch.randn(dims.n_segments, dims.d_graph))
        self.project = nn.Linear(hidden, dims.d_graph, bias=False)
        self.attend = SignedAttention(hidden, attention_dim, graph_heads, negative_slope)
        self.blocks = nn.ModuleList(GraphBiasedBlock(hidden, heads, ff_mult, dropout)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.register_buffer("allowed", steps.unsqueeze(1) >= steps, persistent=False)
        self.output_dim = hidden
        self.shuffle_edges = False
        self.graph_bias = True

    def stable_graph(self) -> torch.Tensor:
        """The stable segment graph, shared by every window.

        Returns
        -------
        torch.Tensor
            Symmetric scores ``[T, T]`` that depend only on parameters.
        """
        scores = self.embed_from @ self.embed_to.T / self.embed_from.shape[1] ** 0.5
        return 0.5 * (scores + scores.T)

    def change(self, nodes: torch.Tensor, stable: torch.Tensor) -> torch.Tensor:
        """How each pair of segments departs from what the stable graph predicts.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment tokens ``[B, T, hidden]``.
        stable : torch.Tensor
            Stable graph scores ``[T, T]``.

        Returns
        -------
        torch.Tensor
            Score changes ``[B, T, T]`` in ``[-1, 1]``; the residual of segment
            ``t`` uses segments ``1..t-1`` only.
        """
        propagate = masked_softmax(stable, self.causal)
        residual = nodes - propagate @ nodes
        projected = F.normalize(self.project(residual), dim=-1)
        scores = projected @ projected.mT
        left = (residual.norm(dim=-1) / nodes.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0)
        return scores * (left.unsqueeze(-1) * left.unsqueeze(-2)).clamp(min=1e-12).sqrt()

    def edge_weights(self, nodes: torch.Tensor) -> torch.Tensor:
        """The signed segment graph ``S^S + Delta`` of every window.

        When ``shuffle_edges`` is true, each window's edges are randomly
        permuted over its earlier -> later pairs (the random-graph control).

        Parameters
        ----------
        nodes : torch.Tensor
            Segment tokens ``[B, T, hidden]``.

        Returns
        -------
        torch.Tensor
            ``[B, T, T]``; entry ``[b, t, s]`` is the edge from ``s`` to ``t``,
            zero where ``s >= t``.
        """
        stable = self.stable_graph()
        weights = (stable + self.change(nodes, stable)) * self.causal
        if self.shuffle_edges:
            values = weights[:, self.causal]
            order = torch.rand_like(values).argsort(-1)
            weights = torch.zeros_like(weights)
            weights[:, self.causal] = values.gather(1, order)
        return weights

    def importance(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed attention over the earlier segments of the stable-plus-change graph.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment tokens ``[B, T, hidden]``.

        Returns
        -------
        tuple of torch.Tensor
            Importance of the positive and of the negative edges ``[B, T, T]``,
            each row summing to one, or to zero without edges of that sign.
        """
        positive, negative = signed_magnitudes(self.edge_weights(nodes), self.causal)
        return self.attend(nodes, positive, negative)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the graph-guided causal Transformer over the segments.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``.
        """
        tokens = self.embed(hidden.flatten(2))
        positive, negative = self.importance(tokens)
        if not self.graph_bias:
            positive = negative = None
        for block in self.blocks:
            tokens = block(tokens, positive, negative, self.allowed)
        return self.norm(tokens)


class LagBiasedBlock(nn.Module):
    """One pre-norm causal Transformer block with a learned bias per lag (Exp 11).

    The attention logit of head ``h`` from segment ``s`` to segment ``t`` is

        q_t . k_s / sqrt(d_head) + b_h(t - s),   s <= t

    with one learned value per head and lag, starting at zero, as in the
    relative position bias of T5 (Raffel et al., JMLR 2020); ALiBi (Press et
    al., ICLR 2022) uses a fixed linear penalty instead. It replaces the
    absolute segment position of Exp 6-10: in a block-aligned window the
    absolute index of a segment means nothing, how long ago it was does, and
    in Exp 10 the absolute position made up 57-68% of every segment token,
    which turned the segment graph into a function of position.

    Parameters
    ----------
    width : int
        Token width.
    heads : int
        Attention heads.
    ff_mult : int
        Width of the feed-forward hidden layer relative to ``width``.
    dropout : float
        Dropout of the attention weights and of both residual branches.
    n_segments : int
        Segments per window, which bounds the lag.
    """

    def __init__(self, width: int, heads: int, ff_mult: int, dropout: float, n_segments: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.lag_bias = nn.Parameter(torch.zeros(heads, n_segments))
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_mult * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_mult * width, width))
        self.drop = nn.Dropout(dropout)
        steps = torch.arange(n_segments)
        self.register_buffer("lag", (steps.unsqueeze(1) - steps).clamp(min=0), persistent=False)
        self.register_buffer("allowed", steps.unsqueeze(1) >= steps, persistent=False)

    def attend(self, normed: torch.Tensor) -> torch.Tensor:
        """Causal self-attention with the lag bias.

        Parameters
        ----------
        normed : torch.Tensor
            Normalised tokens ``[B, T, width]``.

        Returns
        -------
        torch.Tensor
            Attended tokens ``[B, T, width]``, before dropout.
        """
        query, key, value = (self.qkv(normed)
                             .unflatten(-1, (3, self.heads, -1)).permute(2, 0, 3, 1, 4))
        logits = query @ key.mT / query.shape[-1] ** 0.5 + self.lag_bias[:, self.lag]
        weights = self.drop(logits.masked_fill(~self.allowed, float("-inf")).softmax(-1))
        return self.out((weights @ value).transpose(1, 2).flatten(2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Attend over the earlier segments, then the feed-forward layer.

        Parameters
        ----------
        tokens : torch.Tensor
            Segment tokens ``[B, T, width]``.

        Returns
        -------
        torch.Tensor
            Updated tokens ``[B, T, width]``.
        """
        tokens = tokens + self.drop(self.attend(self.norm1(tokens)))
        return tokens + self.drop(self.ff(self.norm2(tokens)))


class SignedGraphBlock(LagBiasedBlock):
    """A lag-biased causal block plus a signed graph attention branch (Exp 11).

    Two branches read the same normalised tokens and their outputs are added
    to the residual stream, as in a GraphGPS layer (Rampasek et al., NeurIPS
    2022), here causal and signed:

    - the global branch is the causal self-attention of :class:`LagBiasedBlock`;
    - the graph branch is one signed graph attention layer (SNEA, Li et al.,
      AAAI 2020) over the segment graph. :class:`SignedAttention`, with
      ``log |G|`` in its logits, weighs the positive and the negative earlier
      segments, and :class:`SignedGraphConvLayer` applies the balance-theory
      update to the two halves of the token: the balanced half gathers from
      positive neighbours' balanced halves and negative neighbours'
      unbalanced halves, and the unbalanced half the other way round.

    In Exp 9b and 10 the graph only added ``beta * I`` to the attention logits,
    where ``I`` is a probability of about ``1 / t``; that bias spread about
    0.1 against 3.5-4 for the query-key logits, so it could not steer the
    attention, and removing it changed nothing. Here the graph branch sends
    its own messages, at the scale of the attention branch.

    Parameters
    ----------
    width : int
        Token width; even, split into balanced and unbalanced halves.
    heads : int
        Attention heads of the global branch.
    ff_mult : int
        Width of the feed-forward hidden layer relative to ``width``.
    dropout : float
        Dropout of the attention weights and of every residual branch.
    n_segments : int
        Segments per window.
    graph_heads : int
        Heads of the signed attention, per sign.
    attention_dim : int
        Width of the attention space of each of those heads.
    negative_slope : float
        Slope of their LeakyReLU.
    """

    def __init__(self, width: int, heads: int, ff_mult: int, dropout: float, n_segments: int,
                 graph_heads: int, attention_dim: int, negative_slope: float):
        super().__init__(width, heads, ff_mult, dropout, n_segments)
        half = width // 2
        self.graph_attend = SignedAttention(width, attention_dim, graph_heads, negative_slope)
        self.signed = SignedGraphConvLayer(half, half, first=False)
        self.graph_out = nn.Linear(width, width)

    def forward(self, tokens: torch.Tensor, positive: torch.Tensor | None,
                negative: torch.Tensor | None) -> torch.Tensor:
        """Global attention plus the signed graph messages, then the feed-forward layer.

        Parameters
        ----------
        tokens : torch.Tensor
            Segment tokens ``[B, T, width]``.
        positive, negative : torch.Tensor or None
            Magnitudes of the positive and negative segment edges ``[B, T, T]``
            from :func:`signed_magnitudes`; ``None`` removes the graph branch.

        Returns
        -------
        torch.Tensor
            Updated tokens ``[B, T, width]``.
        """
        normed = self.norm1(tokens)
        update = self.drop(self.attend(normed))
        if positive is not None:
            to_positive, to_negative = self.graph_attend(normed, positive, negative)
            balanced, unbalanced = normed.chunk(2, -1)
            balanced, unbalanced = self.signed(to_positive, to_negative, balanced, unbalanced)
            update = update + self.drop(self.graph_out(torch.cat([balanced, unbalanced], -1)))
        tokens = tokens + update
        return tokens + self.drop(self.ff(self.norm2(tokens)))


@register("sequence", "lag_transformer")
class LagTransformer(nn.Module):
    """A causal Transformer over the segments with a learned lag bias, no graph (Exp 11a).

    The control of Exp 11: the same tokens, blocks and lag bias as
    :class:`LagGraphTransformer`, without the segment graph and its branch.
    If Exp 11 does not beat it, the segment graph adds nothing.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Token width.
    layers : int
        Transformer blocks.
    heads : int
        Attention heads per block.
    ff_mult : int
        Feed-forward width relative to ``hidden``.
    dropout : float
        Dropout inside the blocks.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, heads: int, ff_mult: int,
                 dropout: float):
        super().__init__()
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.blocks = nn.ModuleList(LagBiasedBlock(hidden, heads, ff_mult, dropout,
                                                   dims.n_segments) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.output_dim = hidden

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the causal Transformer over the segments.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``; segment ``t`` sees
            segments ``1..t``.
        """
        tokens = self.embed(hidden.flatten(2))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


@register("sequence", "lag_graph_transformer")
class LagGraphTransformer(nn.Module):
    """Lag-based segment graph with a signed graph branch in every causal block (Exp 11).

    The segment graph keeps Exp 10's form, ``G_ts = S_ts + Delta_ts`` for
    ``s < t``, with two changes that follow from what Exp 10 learned:

    - ``S`` depends on the lag only, ``S_ts = lag_graph[t - s]``, one learned
      value per lag, starting at zero. Exp 10's ``S^S`` came from random
      embeddings of the absolute segment positions; it kept a correlation of
      0.95-0.97 with its random start, was 95% of every window's graph, and
      every fold started from the same draw. Starting at zero makes the
      window-specific ``Delta`` the whole graph at first. As in layer 1,
      ``S`` also predicts each segment from its earlier ones (softmax over
      them), so it reads as an autoregressive weighting of lags.
    - ``Delta`` is Exp 10's change, computed by the same function
      (:meth:`StableChangeGraphTransformer.change`, reused here as
      :attr:`change`) on tokens that carry no absolute position, so it can
      only reflect the EEG.

    The graph is computed once from the layer-1 tokens and feeds the signed
    graph branch of every :class:`SignedGraphBlock`, next to lag-biased
    causal self-attention. Segment ``t`` sees segments ``1..t``; the head
    reads the last segment and the decoder reads every one. ``graph_bias``
    set to false removes the graph branch from every block, and
    ``shuffle_edges`` places each window's edges at random (both for the
    evaluation); :attr:`edge_weights` is Exp 10's, which reads this
    component's :meth:`stable_graph`.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes; ``d_graph`` is the width the residuals are compared in.
    hidden : int
        Token width.
    layers : int
        Blocks.
    heads : int
        Attention heads of the global branch.
    ff_mult : int
        Feed-forward width relative to ``hidden``.
    dropout : float
        Dropout inside the blocks.
    graph_heads : int
        Heads of the signed attention of the graph branch, per sign.
    attention_dim : int
        Width of the attention space of each of those heads.
    negative_slope : float
        Slope of their LeakyReLU.
    """

    change = StableChangeGraphTransformer.change
    edge_weights = StableChangeGraphTransformer.edge_weights

    def __init__(self, dims: Dimensions, hidden: int, layers: int, heads: int, ff_mult: int,
                 dropout: float, graph_heads: int, attention_dim: int, negative_slope: float):
        super().__init__()
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.lag_graph = nn.Parameter(torch.zeros(dims.n_segments))
        self.project = nn.Linear(hidden, dims.d_graph, bias=False)
        self.blocks = nn.ModuleList(
            SignedGraphBlock(hidden, heads, ff_mult, dropout, dims.n_segments, graph_heads,
                             attention_dim, negative_slope) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.register_buffer("lag", (steps.unsqueeze(1) - steps).clamp(min=0), persistent=False)
        self.output_dim = hidden
        self.shuffle_edges = False
        self.graph_bias = True

    def stable_graph(self) -> torch.Tensor:
        """The lag graph ``S_ts = lag_graph[t - s]``, shared by every window.

        Returns
        -------
        torch.Tensor
            Scores ``[T, T]``, zero where ``s >= t``.
        """
        return self.lag_graph[self.lag] * self.causal

    def importance(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights of the first block's signed graph attention, for the figures.

        Parameters
        ----------
        nodes : torch.Tensor
            Segment tokens ``[B, T, hidden]``, the input of the first block.

        Returns
        -------
        tuple of torch.Tensor
            Weights of the positive and of the negative earlier segments
            ``[B, T, T]``, each row summing to one, or to zero without edges
            of that sign.
        """
        positive, negative = signed_magnitudes(self.edge_weights(nodes), self.causal)
        first = self.blocks[0]
        return first.graph_attend(first.norm1(nodes), positive, negative)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Build the segment graph and run the graph-and-attention blocks.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``.
        """
        tokens = self.embed(hidden.flatten(2))
        positive = negative = None
        if self.graph_bias:
            positive, negative = signed_magnitudes(self.edge_weights(tokens), self.causal)
        for block in self.blocks:
            tokens = block(tokens, positive, negative)
        return self.norm(tokens)


@register("sequence", "window_graph_transformer")
class WindowGraphTransformer(nn.Module):
    """A segment graph computed from the window alone, guiding Exp 11's blocks (Exp 12).

    Exp 11's segment graph ``lag_graph[t - s] + Delta`` became a fixed rule:
    the learned lag values (+2 at lag 1, about -0.5 elsewhere) set the sign of
    93-96% of the edges in every window and stage, and replacing ``Delta`` by
    zero or by another window's left the scores unchanged. Here the graph has
    no learned part at all; every edge is computed from the window:

    1. Each electrode's layer-1 features of a segment are standardised
       (layer norm without parameters) and scaled to unit length.
    2. ``raw_ts`` is the similarity of segments ``t`` and ``s``: the cosine of
       their features, electrode by electrode, averaged over the electrodes.
       It compares the two scalp patterns on the uncompressed ``[C, D]``
       features, as EEG microstate analysis compares topographies, rather
       than on the 128-wide token.
    3. Each row is centred and scaled over the earlier segments ``s < t``:
       ``G_ts = (raw_ts - mean_s raw_ts) / std_s raw_ts``. Centring makes every
       row hold both signs, so the edges cannot collapse to one sign (Exp 8,
       11); the cosine is bounded, so they cannot saturate (Exp 9a, 9b);
       scaling keeps the edges near unit size, so the graph cannot fade when
       the segments become alike. Positive means "more like segment ``t``
       than its average earlier segment", negative "less alike". Segments 1
       and 2 have fewer than two earlier segments and get no edges; they
       still see the past through self-attention. The small constant sits
       inside the square root, because the spread of a row without earlier
       segments is zero and the gradient of ``sqrt`` at zero is infinite.

    The tokens, the lag-biased causal self-attention and the signed graph
    branch of every block are Exp 11's (:class:`SignedGraphBlock`). The edges
    are causal, so segment ``t`` depends on segments ``1..t`` only.

    For the evaluation, ``graph_bias`` set to false removes the graph branch,
    ``shuffle_edges`` places each window's edges at random, and
    ``mean_graph`` (a ``[T, T]`` tensor) replaces every window's graph by
    that one graph, which tests whether the window-specific edges matter.
    ``graph_from_electrodes`` tells the evaluation that :meth:`edge_weights`
    and :meth:`importance` read the layer-1 features, not the tokens.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    hidden : int
        Token width.
    layers : int
        Blocks.
    heads : int
        Attention heads of the global branch.
    ff_mult : int
        Feed-forward width relative to ``hidden``.
    dropout : float
        Dropout inside the blocks.
    graph_heads : int
        Heads of the signed attention of the graph branch, per sign.
    attention_dim : int
        Width of the attention space of each of those heads.
    negative_slope : float
        Slope of their LeakyReLU.
    """

    def __init__(self, dims: Dimensions, hidden: int, layers: int, heads: int, ff_mult: int,
                 dropout: float, graph_heads: int, attention_dim: int, negative_slope: float):
        super().__init__()
        self.embed = nn.Linear(dims.n_channels * dims.d_model, hidden)
        self.blocks = nn.ModuleList(
            SignedGraphBlock(hidden, heads, ff_mult, dropout, dims.n_segments, graph_heads,
                             attention_dim, negative_slope) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        steps = torch.arange(dims.n_segments)
        self.register_buffer("causal", steps.unsqueeze(1) > steps, persistent=False)
        self.output_dim = hidden
        self.shuffle_edges = False
        self.graph_bias = True
        self.mean_graph: torch.Tensor | None = None
        self.graph_from_electrodes = True

    def similarity(self, hidden: torch.Tensor) -> torch.Tensor:
        """Similarity of the scalp patterns of every pair of segments.

        Parameters
        ----------
        hidden : torch.Tensor
            Layer-1 node representations ``[B, T, C, D]``.

        Returns
        -------
        torch.Tensor
            ``[B, T, T]`` in ``[-1, 1]``: the per-electrode cosine of the
            standardised features of two segments, averaged over electrodes.
        """
        features = F.normalize(F.layer_norm(hidden, hidden.shape[-1:]), dim=-1)
        return torch.einsum("btcd,bscd->bts", features, features) / hidden.shape[2]

    def edge_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        """The signed segment graph of every window: row-standardised similarity.

        Parameters
        ----------
        hidden : torch.Tensor
            Layer-1 node representations ``[B, T, C, D]``.

        Returns
        -------
        torch.Tensor
            ``[B, T, T]``; entry ``[b, t, s]`` is the edge from ``s`` to ``t``,
            zero where ``s >= t`` and in rows with fewer than two earlier
            segments.
        """
        raw = self.similarity(hidden)
        causal = self.causal.to(raw.dtype)
        count = causal.sum(-1, keepdim=True)
        mean = (raw * causal).sum(-1, keepdim=True) / count.clamp(min=1)
        spread = (((raw - mean) * causal) ** 2).sum(-1, keepdim=True) / count.clamp(min=1)
        weights = (raw - mean) / (spread + 1e-6).sqrt() * causal * (count >= 2)
        if self.mean_graph is not None:
            weights = self.mean_graph.to(weights).expand_as(weights)
        if self.shuffle_edges:
            values = weights[:, self.causal]
            order = torch.rand_like(values).argsort(-1)
            weights = torch.zeros_like(weights)
            weights[:, self.causal] = values.gather(1, order)
        return weights

    def importance(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights of the first block's signed graph attention, for the figures.

        Parameters
        ----------
        hidden : torch.Tensor
            Layer-1 node representations ``[B, T, C, D]``.

        Returns
        -------
        tuple of torch.Tensor
            Weights of the positive and of the negative earlier segments
            ``[B, T, T]``, each row summing to one, or to zero without edges
            of that sign.
        """
        positive, negative = signed_magnitudes(self.edge_weights(hidden), self.causal)
        first = self.blocks[0]
        return first.graph_attend(first.norm1(self.embed(hidden.flatten(2))), positive, negative)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute the window's segment graph and run the graph-and-attention blocks.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]`` from layer 1.

        Returns
        -------
        torch.Tensor
            One output per segment, ``[B, T, hidden]``.
        """
        tokens = self.embed(hidden.flatten(2))
        positive = negative = None
        if self.graph_bias:
            positive, negative = signed_magnitudes(self.edge_weights(hidden), self.causal)
        for block in self.blocks:
            tokens = block(tokens, positive, negative)
        return self.norm(tokens)


@register("decoder", "gru_continuous")
class ContinuousGRUDecoder(nn.Module):
    """Rebuild each electrode as one continuous 1000-sample sequence (Exp 4+).

    The node representation plus the projected sequence context of its
    segment is repeated for every sample, and a GRU whose state crosses
    segment boundaries turns it into one amplitude per sample.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    context_dim : int
        Width of the sequence model output.
    sample_position : bool
        Add a sinusoidal encoding of each sample's position inside its
        segment to the decoder input (Exp 6a).
    """

    def __init__(self, dims: Dimensions, context_dim: int, sample_position: bool):
        super().__init__()
        self.segment_len = dims.segment_len
        self.context = nn.Linear(context_dim, dims.d_model)
        self.rnn = nn.GRU(dims.d_model, dims.d_model, batch_first=True)
        self.out = nn.Linear(dims.d_model, 1)
        self.sample_position = sample_position
        if sample_position:
            self.register_buffer("table", sinusoidal_positions(dims.segment_len, dims.d_model),
                                 persistent=False)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Reconstruct the waveform of every electrode and segment.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.
        context : torch.Tensor
            Sequence model outputs ``[B, T, H]``.

        Returns
        -------
        torch.Tensor
            Reconstructed segments ``[B, T, C, L]``.
        """
        latent = hidden + self.context(context).unsqueeze(2)
        batch_size, n_segments, n_channels, _ = latent.shape
        latent = latent.permute(0, 2, 1, 3).repeat_interleave(self.segment_len, dim=2)
        latent = latent.flatten(0, 1)
        if self.sample_position:
            latent = latent + self.table.repeat(n_segments, 1)
        decoded, _ = self.rnn(latent)
        rebuilt = self.out(decoded).squeeze(-1)
        rebuilt = rebuilt.unflatten(1, (n_segments, self.segment_len))
        rebuilt = rebuilt.unflatten(0, (batch_size, n_channels))
        return rebuilt.permute(0, 2, 1, 3)


@register("decoder", "mlp_segment")
class SegmentMLPDecoder(nn.Module):
    """Rebuild each segment directly from its latent, with no recurrence (Exp 15).

    :class:`ContinuousGRUDecoder` runs one GRU over the whole window, so a
    30 s epoch at 1 kHz is a 30,000-step recurrence on an input held
    constant within each segment: the GRU generates the waveform by its own
    dynamics. The spectral loss rewards oscillating dynamics, the regime in
    which gradients through time diverge (Mikhaeil et al., NeurIPS 2022);
    in Exp 14a they reached norms of 1e15 to 1e27, overflowed float32 in the
    norm and made most stage-1 steps of two folds non-finite. A GRU restarted
    at every segment from its latent (1000 steps) made this rarer but not
    impossible. This decoder has no recurrence and no backpropagation
    through time: a two-layer MLP maps the layer-normalised latent of an
    electrode and segment to its ``segment_len`` samples, so the gradient of
    a sample passes through two layers whatever the window length. The
    history the continuous state carried reaches it through the latent: the
    node representation holds the accumulated change ``Delta_t`` of segments
    ``1..t`` and the sequence context is causal. The decoder is used in
    stage 1 only, so its size does not change the classifier.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    context_dim : int
        Width of the sequence model output.
    hidden : int
        Width of the hidden layer.
    """

    def __init__(self, dims: Dimensions, context_dim: int, hidden: int):
        super().__init__()
        self.context = nn.Linear(context_dim, dims.d_model)
        self.norm = nn.LayerNorm(dims.d_model)
        self.net = nn.Sequential(nn.Linear(dims.d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, dims.segment_len))

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Reconstruct the waveform of every electrode and segment.

        Parameters
        ----------
        hidden : torch.Tensor
            Node representations ``[B, T, C, D]``.
        context : torch.Tensor
            Sequence model outputs ``[B, T, H]``.

        Returns
        -------
        torch.Tensor
            Reconstructed segments ``[B, T, C, L]``.
        """
        return self.net(self.norm(hidden + self.context(context).unsqueeze(2)))


@register("head", "mlp")
class MLPHead(nn.Sequential):
    """LayerNorm and a two-hidden-layer GELU MLP (Exp 3+).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    in_dim : int
        Width of the sequence summary.
    hidden : int
        Width of the first hidden layer; the second is half of it.
    dropout : float
        Dropout after each hidden layer.
    """

    def __init__(self, dims: Dimensions, in_dim: int, hidden: int, dropout: float):
        super().__init__(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, dims.n_classes),
        )


@register("head", "linear")
class LinearHead(nn.Linear):
    """A single linear layer (Exp 1-2).

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    in_dim : int
        Width of the sequence summary.
    """

    def __init__(self, dims: Dimensions, in_dim: int):
        super().__init__(in_dim, dims.n_classes)


@register("head", "residual_mlp")
class ResidualMLPHead(nn.Module):
    """Pre-norm residual MLP blocks, then the two-layer output of :class:`MLPHead` (Exp 12).

    Each block is ``x + Dropout(Linear(GELU(Linear(LayerNorm(x)))))``, the
    feed-forward block of a pre-norm Transformer, so extra depth starts close
    to the identity and does not make the head harder to train. The output
    layers are ``LayerNorm -> Linear(in_dim, in_dim / 2) -> GELU -> Dropout ->
    Linear(in_dim / 2, n_classes)``.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes.
    in_dim : int
        Width of the sequence summary.
    hidden : int
        Width inside each residual block.
    blocks : int
        Residual blocks.
    dropout : float
        Dropout inside every block and before the last layer.
    """

    def __init__(self, dims: Dimensions, in_dim: int, hidden: int, blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden, in_dim), nn.Dropout(dropout))
            for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim // 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(in_dim // 2, dims.n_classes))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Class logits from the sequence summary.

        Parameters
        ----------
        summary : torch.Tensor
            ``[B, in_dim]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, n_classes]``.
        """
        for block in self.blocks:
            summary = summary + block(summary)
        return self.out(summary)


@register("readout", "last")
class LastReadout(nn.Module):
    """The output of the last segment, as every experiment before Exp 13 read it.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes (unused).
    in_dim : int
        Width of the sequence outputs.
    """

    def __init__(self, dims: Dimensions, in_dim: int):
        super().__init__()
        self.output_dim = in_dim

    def forward(self, outputs: torch.Tensor, elapsed: torch.Tensor | None) -> torch.Tensor:
        """Take the last segment.

        Parameters
        ----------
        outputs : torch.Tensor
            Sequence outputs ``[B, T, in_dim]``.
        elapsed : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``[B, in_dim]``.
        """
        return outputs[:, -1]


@register("readout", "attention_pool")
class AttentionPoolReadout(nn.Module):
    """Attention pooling over the last segments, optionally with the elapsed time (Exp 13).

    With rolling windows the label is the stage of the window's last second,
    so the readout pools the outputs of the segments in that second instead
    of reading only the last segment: one learned score per segment, a
    softmax over the last ``tokens`` segments and their weighted sum, as the
    classifier of MTSSRL-MD (ICLR 2026 submission) pools its features. The
    score layer starts at zero, so pooling starts as a plain average. With
    ``elapsed_time`` the standardised time since the recording started is
    appended, a counter that a live system also has.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes (unused).
    in_dim : int
        Width of the sequence outputs.
    tokens : int
        Last segments pooled.
    elapsed_time : bool
        Append the elapsed time to the pooled vector.
    """

    def __init__(self, dims: Dimensions, in_dim: int, tokens: int, elapsed_time: bool):
        super().__init__()
        self.tokens, self.elapsed_time = tokens, elapsed_time
        self.score = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.output_dim = in_dim + int(elapsed_time)

    def forward(self, outputs: torch.Tensor, elapsed: torch.Tensor | None) -> torch.Tensor:
        """Pool the last segments and append the elapsed time.

        Parameters
        ----------
        outputs : torch.Tensor
            Sequence outputs ``[B, T, in_dim]``.
        elapsed : torch.Tensor or None
            Standardised elapsed time ``[B]``; required when ``elapsed_time``
            is set.

        Returns
        -------
        torch.Tensor
            ``[B, output_dim]``.

        Raises
        ------
        ValueError
            If the elapsed time is needed but missing.
        """
        last = outputs[:, -self.tokens:]
        weights = self.score(last).softmax(1)
        pooled = (weights * last).sum(1)
        if not self.elapsed_time:
            return pooled
        if elapsed is None:
            raise ValueError("this readout needs the elapsed time; use rolling windows")
        return torch.cat([pooled, elapsed.to(pooled).unsqueeze(-1)], -1)



@register("context", "none")
class NoContext(nn.Module):
    """Classify each window from itself alone, as every experiment before Exp 14b did.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes (unused).
    in_dim : int
        Width of the readout vectors, passed through unchanged.
    """

    def __init__(self, dims: Dimensions, in_dim: int):
        super().__init__()
        self.output_dim = in_dim

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Return the readout vectors unchanged.

        Parameters
        ----------
        summary : torch.Tensor
            ``[B, in_dim]`` or, for runs of epochs, ``[B, N, in_dim]``.

        Returns
        -------
        torch.Tensor
            ``summary`` itself.
        """
        return summary


@register("context", "gru")
class EpochContextGRU(nn.Module):
    """A one-direction GRU over the readout vectors of consecutive epochs (Exp 14b).

    Scorers stage an epoch with its neighbours in view, and the AASM rules
    refer to the epochs before it (an ambiguous epoch after REM stays REM; N1
    mostly follows wake). The GRU reads the epochs of an item in order, so
    the output at epoch ``n`` depends on epochs ``1..n`` of the item only,
    which keeps the model causal at the epoch level as the segment GRU and
    the causal blocks are within an epoch. Sequence-to-sequence reading of
    epoch features is established (SeqSleepNet, Phan et al., IEEE TNSRE
    2019; TinySleepNet, Supratak & Guo, EMBC 2020); here it is one-directional.
    A single window is read as a run of one epoch.

    Parameters
    ----------
    dims : Dimensions
        Shared sizes (unused).
    in_dim : int
        Width of the readout vectors.
    hidden : int
        Hidden width.
    layers : int
        Stacked layers.
    dropout : float
        Dropout between stacked layers.
    """

    def __init__(self, dims: Dimensions, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.output_dim = hidden

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Run over the epochs of each item.

        Parameters
        ----------
        summary : torch.Tensor
            ``[B, N, in_dim]`` for runs of epochs, or ``[B, in_dim]``.

        Returns
        -------
        torch.Tensor
            ``[B, N, hidden]``, or ``[B, hidden]`` for single windows.
        """
        if summary.dim() == 2:
            return self.rnn(summary.unsqueeze(1))[0].squeeze(1)
        return self.rnn(summary)[0]
