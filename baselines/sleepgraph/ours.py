"""Our configuration for the sleep-graph-benchmark models: our data and trainer, their networks.

The networks are upstream's classes, imported from ``upstream/models`` (that
package imports only torch, so it can live beside our modules; upstream's
``data`` package is never imported here). What our configuration changes is
only what our data force:

- **Input:** our runs of ``data.context_epochs`` scaled 1 kHz epochs of the 83
  electrodes, from :func:`baselines.common.ours.our_loaders`.
- **Window:** each model keeps its own window length (5 or 9 epochs), but the
  window ends at the epoch it labels and uses only that epoch and earlier ones
  of the run (repeating the run's first epoch where the run is shorter), as our
  protocol is causal; upstream's windows are centred on the labelled epoch.
  Every epoch of a run gets its own window, so a run gives one logit row per
  epoch, as :func:`train.classify` expects.
- **Graph of the labelled epoch:** GraphSleepNet builds its graph from the
  window's centre epoch; here from the labelled (last) epoch, so the graph
  still belongs to the epoch being classified.
- **Training:** our stage 2 optimises the classification loss only, so
  upstream's auxiliary graph losses (ProductGraphSleepNet's smoothness term,
  MSTGCN's two graph terms) and MSTGCN's domain-adversarial head are not used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baselines.common.ours import first_items, item_keys, merge_nested, our_loaders, restrict_table
from baselines.common.ours import OurProtocolModel
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from evaluate import predict_proba
from train import classify

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"


def upstream_models():
    """Import upstream's ``models`` package, with our modules taking precedence.

    The repository goes at the end of ``sys.path``: our ``data`` module is
    found first, and upstream's ``models`` (a name we do not use) is found
    there.

    Returns
    -------
    ModuleType
        ``models``, with ``graphsleepnet``, ``productgraphsleepnet``,
        ``mstgcn`` and ``encoders`` imported.
    """
    if str(UPSTREAM) not in sys.path:
        sys.path.append(str(UPSTREAM))
    import models.encoders
    import models.graphsleepnet
    import models.mstgcn
    import models.productgraphsleepnet
    return models


def causal_windows(features: torch.Tensor, window: int) -> torch.Tensor:
    """For every epoch of a run, the window of ``window`` epochs that ends at it.

    Parameters
    ----------
    features : torch.Tensor
        Per-epoch node features ``[B, S, V, F]``.
    window : int
        Epochs per window.

    Returns
    -------
    torch.Tensor
        ``[B * S, window, V, F]``; window ``s`` holds epochs ``s - window + 1 .. s``,
        the run's first epoch repeated where the run is shorter.
    """
    steps = features.shape[1]
    index = (torch.arange(steps, device=features.device)[:, None]
             + torch.arange(1 - window, 1, device=features.device)[None, :]).clamp(min=0)
    return features[:, index].flatten(0, 1)


def distance_adjacency(coordinates: np.ndarray, sigma: float) -> np.ndarray:
    """The benchmark's distance graph, built from our montage coordinates.

    Upstream's generator (``models/mstgcn.py:500-530``) places the 10-20
    electrodes on 2-D coordinates of a unit head, ``A = exp(-d^2 / (2 sigma^2))``
    with ``sigma = 0.35``, keeps the self-loops and normalises each row. It
    knows few of our 83 electrode names, so here the 3-D montage positions
    are centred and scaled to a mean radius of one and the same kernel is
    applied to their distances.

    Parameters
    ----------
    coordinates : np.ndarray
        ``[V, 3]`` montage positions.
    sigma : float
        Kernel width in head radii.

    Returns
    -------
    np.ndarray
        Row-normalised ``[V, V]`` adjacency.
    """
    centred = coordinates - coordinates.mean(0)
    centred = centred / np.linalg.norm(centred, axis=1).mean()
    distance = np.linalg.norm(centred[:, None] - centred[None], axis=-1)
    adjacency = np.exp(-distance ** 2 / (2.0 * sigma ** 2))
    return (adjacency / adjacency.sum(1, keepdims=True)).astype(np.float32)


class DifferentialEntropy(nn.Module):
    """Upstream's nine-band differential entropy with a z-score fitted on training runs.

    Parameters
    ----------
    sampling_rate : float
        Hz of our epochs.
    n_channels : int
        Electrodes.
    """

    def __init__(self, sampling_rate: float, n_channels: int):
        super().__init__()
        self.extract = upstream_models().encoders.OfficialDifferentialEntropy(sampling_rate)
        self.register_buffer("mean", torch.zeros(n_channels, 9))
        self.register_buffer("std", torch.ones(n_channels, 9))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Differential entropy of every epoch, before the z-score.

        Parameters
        ----------
        x : torch.Tensor
            ``[..., V, samples]``.

        Returns
        -------
        torch.Tensor
            ``[..., V, 9]``, computed in float32.
        """
        return self.extract(x.float())

    @torch.no_grad()
    def fit(self, loader, n_batches: int | None, device: torch.device) -> int:
        """Per-electrode, per-band mean and SD over the epochs of training runs.

        Upstream z-scores over every epoch of the training nights; here over
        every epoch of the runs ``loader`` yields (the unaugmented random
        training runs of ``train_eval``).

        Parameters
        ----------
        loader : DataLoader
            Training runs ``[B, S, V, samples]``.
        n_batches : int or None
            Batches to read; None reads all.
        device : torch.device
            Device of the computation.

        Returns
        -------
        int
            Epochs used.
        """
        total, square, count = 0.0, 0.0, 0
        for index, (x, _) in enumerate(loader):
            if n_batches is not None and index >= n_batches:
                break
            values = self.raw(x.to(device)).double().flatten(0, -3)
            total = total + values.sum(0)
            square = square + values.pow(2).sum(0)
            count += values.shape[0]
        mean = total / count
        self.mean.copy_(mean.float())
        self.std.copy_((square / count - mean.pow(2)).clamp(min=0).sqrt().float().clamp(min=1e-6))
        return count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Z-scored differential entropy.

        Parameters
        ----------
        x : torch.Tensor
            ``[..., V, samples]``.

        Returns
        -------
        torch.Tensor
            ``[..., V, 9]``.
        """
        return (self.raw(x) - self.mean) / self.std


def run_graph_ours(network: nn.Module, settings: dict, overrides: dict, out_dir: Path,
                   quick: bool, de_fit_batches: int) -> dict:
    """Train and score a graph baseline under our configuration, and keep its test graphs.

    The same steps as :func:`baselines.common.ours.run_ours`; in addition the
    network's differential-entropy z-score is fitted first (networks with a
    ``features.fit``), and the graph of every scored test epoch is saved to
    ``graphs.npz``.

    Parameters
    ----------
    network : nn.Module
        Wrapper with ``classifier`` and ``last_graph`` (the ``[B * S, V, V]``
        graphs of its last forward pass).
    settings : dict
        Contents of ``baselines/settings.yaml``.
    overrides : dict
        Nested settings merged on the reference configuration.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test.
    de_fit_batches : int
        Batches of the training loader used for the z-score in smoke tests,
        where ``train_eval`` is not built.

    Returns
    -------
    dict
        Scores, parameter count and the number of epochs behind the z-score.
    """
    quick_settings = settings["quick"]
    if quick:
        overrides = merge_nested(overrides, {"finetune": {"epochs": quick_settings["epochs"]},
                                             "evaluation": {"curve_batches": 0}})
    cfg = load_reference_config(Path(settings["reference_run"]), overrides)
    split = load_fold(Path(settings["fold"]))
    if quick:
        split = {name: split[name][:quick_settings[f"{name}_subjects"]] for name in split}
    loaders = our_loaders(cfg, split)
    if quick:
        for name in ("val", "test"):
            loaders[name] = first_items(loaders[name], quick_settings[f"{name}_items"])
    out_dir.mkdir(parents=True, exist_ok=True)
    network = network.to(cfg.device)
    fitted = None
    if hasattr(network, "features") and hasattr(network.features, "fit"):
        source = loaders.get("train_eval")
        fitted = network.features.fit(source if source is not None else loaders["train"],
                                      None if source is not None else de_fit_batches, cfg.device)
    model = OurProtocolModel(network).to(cfg.device)
    parameters = sum(p.numel() for p in model.parameters())
    history, best = classify(cfg, model, loaders["train"], loaders["val"],
                             out_dir / "classify_best.pt", loaders.get("train_eval"))
    _, probability = predict_proba(cfg, model, loaders["test"])
    names, onsets = item_keys(loaders["test"], split["test"])
    graphs = test_graphs(cfg, model, loaders["test"])
    np.savez_compressed(out_dir / "graphs.npz", subject=np.array(names), onset=np.array(onsets),
                        graph=graphs)
    path = out_dir / "predictions.npz"
    save_predictions(path, np.array(names), np.array(onsets), probability.argmax(-1), probability)
    table = scored_epochs(cfg, split["test"])
    if quick:
        table = restrict_table(table, names, onsets)
    result = score_run(cfg, table, path)
    result.update({"configuration": "ours", "parameters": parameters, "val_best_macro_f1": best,
                   "quick": quick, "de_fit_epochs": fitted, "history": history})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


@torch.no_grad()
def test_graphs(cfg, model: nn.Module, loader) -> np.ndarray:
    """The electrode graph of every scored test epoch, in the loader's order.

    Parameters
    ----------
    cfg : Config
        Reference configuration (device).
    model : nn.Module
        Trained :class:`OurProtocolModel`.
    loader : DataLoader
        Test items, one labelled epoch each.

    Returns
    -------
    np.ndarray
        ``[n_items, V, V]`` float16.
    """
    model.eval()
    graphs = []
    for x, y in loader:
        model(x.to(cfg.device))
        steps = y.shape[1]
        position = (y >= 0).float().argmax(1)
        last = model.network.last_graph.view(y.shape[0], steps, *model.network.last_graph.shape[1:])
        graphs.append(last[torch.arange(y.shape[0]), position].float().cpu().numpy())
    return np.concatenate(graphs).astype(np.float16)
