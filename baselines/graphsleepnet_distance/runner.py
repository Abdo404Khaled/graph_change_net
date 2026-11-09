from __future__ import annotations

from pathlib import Path

import torch

import data
from baselines.common.reference import load_reference_config
from baselines.graphsleepnet.model import GraphSleepNetRuns
from baselines.sleepgraph.native import run_native
from baselines.sleepgraph.ours import distance_adjacency, run_graph_ours

ROOT = Path(__file__).resolve().parent


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Train and score GraphSleepNet with a fixed distance graph under one configuration.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml`` or ``ours.yaml``.
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
        If the configuration is neither ``native`` nor ``ours``.
    """
    if spec["configuration"] == "native":
        return run_native("graphsleepnet_distance", spec, settings, out_dir, quick,
                          launcher=ROOT / "native_launcher.py",
                          distance=out_dir / "upstream" / "distance_adjacency.npy")
    if spec["configuration"] == "ours":
        torch.manual_seed(spec["seed"])
        cfg = load_reference_config(Path(settings["reference_run"]))
        channels, coordinates = data.read_montage(data.find_montage(cfg.paths.data_root))
        network = GraphSleepNetRuns(len(channels), cfg.data.sfreq, spec["window"],
                                    distance_adjacency(coordinates, spec["sigma"]))
        return run_graph_ours(network, settings, spec["overrides"], out_dir, quick,
                              spec["de_fit_batches"])
    raise ValueError(f"unknown configuration {spec['configuration']!r}")
