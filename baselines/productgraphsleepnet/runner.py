from __future__ import annotations

from pathlib import Path

import torch

import data
from baselines.common.reference import load_reference_config
from baselines.productgraphsleepnet.model import ProductGraphSleepNetRuns
from baselines.sleepgraph.native import run_native
from baselines.sleepgraph.ours import run_graph_ours


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Train and score ProductGraphSleepNet under one configuration.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``, ``cnn2s.yaml``, ``causal.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        Scores and parameter count.

    Raises
    ------
    ValueError
        If the configuration is not ``native``, ``cnn2s``, ``causal`` or ``ours``.
    """
    if spec["configuration"] in ("native", "cnn2s", "causal"):
        return run_native("productgraphsleepnet", spec, settings, out_dir, quick)
    if spec["configuration"] == "ours":
        torch.manual_seed(spec["seed"])
        cfg = load_reference_config(Path(settings["reference_run"]))
        channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
        network = ProductGraphSleepNetRuns(len(channels), cfg.data.sfreq, spec["window"],
                                           spec["backbone"])
        return run_graph_ours(network, settings, spec["overrides"], out_dir, quick,
                              spec["de_fit_batches"])
    raise ValueError(f"unknown configuration {spec['configuration']!r}")
