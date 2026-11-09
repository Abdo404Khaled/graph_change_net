from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import data
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from config import Config
from evaluate import predict_proba
from train import classify


def our_loaders(cfg: Config, split: dict[str, list[str]]) -> dict[str, DataLoader]:
    """Our data protocol, unchanged, for a model that is not ours.

    The loaders are built by :func:`data.build_epoch_loaders`, so the data are
    exactly what the reference model saw: the same cache (1 kHz, never
    resampled; notch, 0.5-45 Hz band-pass, common average over the 83 montage
    electrodes), the per-electrode robust scaler fitted on the training
    subjects and clipped to +-20, runs of ``data.context_epochs`` epochs, the
    training augmentations, artifact epochs dropped from training only, and
    one validation or test item per scored epoch that ends at it.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    split : dict
        Subject ids per split.

    Returns
    -------
    dict
        The loaders of :func:`data.build_epoch_loaders`.

    Raises
    ------
    ValueError
        If a split subject has no data.
    """
    arrays, _, _ = data.load_dataset(cfg)
    missing = sorted({s for subjects in split.values() for s in subjects} - set(arrays))
    if missing:
        raise ValueError(f"split subjects without data: {missing}")
    return data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)


def first_items(loader: DataLoader, n_items: int) -> DataLoader:
    """The same loader restricted to its first items, for smoke tests.

    Parameters
    ----------
    loader : DataLoader
        An unshuffled loader.
    n_items : int
        Items to keep.

    Returns
    -------
    DataLoader
        Loader over the first ``n_items`` items with the same batch size.
    """
    subset = Subset(loader.dataset, range(min(n_items, len(loader.dataset))))
    return DataLoader(subset, loader.batch_size, num_workers=loader.num_workers)


def item_keys(loader: DataLoader, subjects: list[str]) -> tuple[list[str], list[float]]:
    """Night and onset of the labelled epoch of every item of a val or test loader.

    :class:`data.EpochDataset` stores, per item, the position of its night in
    the subject list it was built from and the row of its labelled epoch, so
    the identity of each prediction is recovered without changing that class.

    Parameters
    ----------
    loader : DataLoader
        A ``val`` or ``test`` loader of :func:`our_loaders` (not shuffled),
        possibly restricted by :func:`first_items`.
    subjects : list of str
        The subject list the loader was built from.

    Returns
    -------
    names : list of str
        Subject id per item.
    onsets : list of float
        Onset in seconds of the labelled epoch of each item.
    """
    dataset = loader.dataset
    indices = list(dataset.indices) if isinstance(dataset, Subset) else range(len(dataset))
    dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    names, onsets = [], []
    for item in indices:
        index, row = int(dataset.subject[item]), int(dataset.row[item])
        names.append(subjects[index])
        onsets.append(float(dataset.epochs[index]["onset"][row]))
    return names, onsets


class OurProtocolModel(nn.Module):
    """A baseline network presented with the interface our stage-2 trainer expects.

    :func:`train.classify` calls the model on runs of epochs and expects
    ``(logits, delta, extra)``, and its optimiser treats ``head``,
    ``readout`` and ``context`` as the head and everything else as the
    backbone. The baseline's final classification layer is the head; the
    readout and context are empty, and the change term is a zero, so
    ``finetune.delta_penalty`` adds nothing.

    Parameters
    ----------
    network : nn.Module
        Maps ``[B, S, C, T]`` to logits ``[B, S, n_stages]`` and has a
        ``classifier`` attribute holding its final classification layer.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.head = network.classifier
        self.readout = nn.Identity()
        self.context = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Stage logits of every epoch of every run.

        Parameters
        ----------
        x : torch.Tensor
            Runs of epochs ``[B, S, C, T]``.

        Returns
        -------
        logits : torch.Tensor
            ``[B, S, n_stages]``.
        delta : torch.Tensor
            A zero, standing in for the graph change.
        extra : None
            Unused.
        """
        logits = self.network(x)
        return logits, logits.new_zeros(1), None


def run_ours(network: nn.Module, settings: dict, overrides: dict, out_dir: Path,
             quick: bool) -> dict:
    """Train and score a baseline network under our configuration, unchanged.

    The data come from :func:`our_loaders` and the training is our stage 2,
    :func:`train.classify`, with the reference run's settings; ``overrides``
    may only change what the baseline's situation requires (for example no
    frozen probe for a network trained from scratch, as in Exp 15a). The
    test predictions are scored by :func:`baselines.common.scoring.score_run`.

    Parameters
    ----------
    network : nn.Module
        Baseline network (see :class:`OurProtocolModel`).
    settings : dict
        Contents of ``baselines/settings.yaml``.
    overrides : dict
        Nested settings merged on the reference configuration.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test: a few subjects, items and epochs (``settings["quick"]``),
        scored on the predicted items only.

    Returns
    -------
    dict
        The scores, with the parameter count.
    """
    quick_settings = settings["quick"]
    overrides = {**overrides}
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
    model = OurProtocolModel(network).to(cfg.device)
    parameters = sum(p.numel() for p in model.parameters())
    history, best = classify(cfg, model, loaders["train"], loaders["val"],
                             out_dir / "classify_best.pt", loaders.get("train_eval"))
    _, probability = predict_proba(cfg, model, loaders["test"])
    names, onsets = item_keys(loaders["test"], split["test"])
    path = out_dir / "predictions.npz"
    save_predictions(path, np.array(names), np.array(onsets), probability.argmax(-1), probability)
    table = scored_epochs(cfg, split["test"])
    if quick:
        table = restrict_table(table, names, onsets)
    result = score_run(cfg, table, path)
    result.update({"parameters": parameters, "val_best_macro_f1": best, "quick": quick,
                   "history": history})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def restrict_table(table: dict[str, dict[str, np.ndarray]], names: list[str],
                   onsets: list[float]) -> dict[str, dict[str, np.ndarray]]:
    """Keep only the reference epochs that were predicted (smoke tests only).

    Parameters
    ----------
    table : dict
        Reference epochs per night.
    names, onsets : list
        Night and onset of each predicted epoch.

    Returns
    -------
    dict
        The same table restricted to the predicted epochs.
    """
    predicted = {(name, round(onset, 3)) for name, onset in zip(names, onsets)}
    restricted = {}
    for name, epochs in table.items():
        keep = np.array([(name, round(onset, 3)) in predicted for onset in epochs["onset"]])
        if keep.any():
            restricted[name] = {key: values[keep] for key, values in epochs.items()}
    return restricted


def merge_nested(base: dict, override: dict) -> dict:
    """Merge two nested dictionaries, the second winning.

    Parameters
    ----------
    base, override : dict
        Nested settings.

    Returns
    -------
    dict
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        merged[key] = (merge_nested(merged[key], value)
                       if isinstance(value, dict) and isinstance(merged.get(key), dict) else value)
    return merged
