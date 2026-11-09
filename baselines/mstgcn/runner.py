from __future__ import annotations

from pathlib import Path

import torch

import data
from baselines.common.reference import load_reference_config
from baselines.mstgcn.model import MSTGCNRuns
from baselines.sleepgraph.native import run_native
from baselines.sleepgraph.ours import distance_adjacency, run_graph_ours


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Train and score MSTGCN under one configuration.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``, ``causal.yaml`` or ``ours.yaml``.
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
        If the configuration is not ``native``, ``causal`` or ``ours``.
    """
    if spec["configuration"] in ("native", "causal"):
        return run_native("mstgcn", spec, settings, out_dir, quick)
    if spec["configuration"] == "ours":
        torch.manual_seed(spec["seed"])
        cfg = load_reference_config(Path(settings["reference_run"]))
        channels, coordinates = data.read_montage(data.find_montage(cfg.paths.data_root))
        network = MSTGCNRuns(len(channels), spec["feature_net"], spec["window"],
                             distance_adjacency(coordinates, spec["sigma"]))
        return run_graph_ours(network, settings, spec["overrides"], out_dir, quick,
                              spec["de_fit_batches"])
    raise ValueError(f"unknown configuration {spec['configuration']!r}")
