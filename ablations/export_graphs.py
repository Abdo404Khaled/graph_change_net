"""Export the electrode graphs of a finished GraphChangeNet run for the graph tests.

``python -m ablations.export_graphs <run folder>`` (Exp 15's ``runs/exp15_final_fold<k>``
or an ablation's ``ablations/results/<name>_fold<k>``) rebuilds the run's test
loader from its own ``config.yaml`` and split, loads ``classify_best.pt`` and
writes ``graphs.npz`` into the run folder, in the format of the graph
baselines (``baselines/graph_tests.py``):

- ``graph``: for every scored test epoch, the mean of its 30 segment graphs
  ``L^S + Delta_t`` (whatever the run's graph learner returns), so that our
  model is compared with the one-graph-per-epoch baselines on the same footing;
- ``delta_norm``: the Frobenius norm of each segment's change ``Delta_t``,
  ``[N, 30]``, for the within-epoch tests;
- ``stable``: the stable graph ``[C, C]``; ``subject`` and ``onset``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import data
from baselines.common.ours import item_keys
from config import load_run_config
from model import GraphChangeNet, load_weights


@torch.no_grad()
def export(run_dir: Path) -> Path:
    """Write ``graphs.npz`` for one run.

    Parameters
    ----------
    run_dir : Path
        A finished run with ``config.yaml``, ``metrics.json`` and
        ``classify_best.pt``.

    Returns
    -------
    Path
        The written file.
    """
    cfg = load_run_config(run_dir, {})
    split = json.loads((run_dir / "metrics.json").read_text())["split"]
    arrays, channels, coordinates = data.load_dataset(cfg)
    loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])},
                                       split)
    model = GraphChangeNet(cfg, len(channels), coordinates).to(cfg.device)
    load_weights(model, run_dir / "classify_best.pt", cfg.device)
    model.eval()
    captured = {}
    model.graph_learner.register_forward_hook(
        lambda module, inputs, output: captured.update(scores=output[0], delta=output[1]))
    graphs, delta_norms = [], []
    for x, y in loaders["test"]:
        model(x.to(cfg.device))
        steps = y.shape[1]
        position = (y >= 0).float().argmax(1).to(cfg.device)
        rows = torch.arange(y.shape[0], device=cfg.device)
        scores = captured["scores"].unflatten(0, (y.shape[0], steps))[rows, position]
        delta = captured["delta"].unflatten(0, (y.shape[0], steps))[rows, position]
        graphs.append(scores.mean(1).float().cpu().numpy())
        delta_norms.append(delta.flatten(-2).norm(dim=-1).float().cpu().numpy())
    names, onsets = item_keys(loaders["test"], split["test"])
    stable = model.graph_learner(torch.zeros(1, cfg.data.n_segments, len(channels),
                                             cfg.data.segment_len, device=cfg.device))[2]
    path = run_dir / "graphs.npz"
    np.savez_compressed(path, subject=np.array(names), onset=np.array(onsets),
                        graph=np.concatenate(graphs).astype(np.float16),
                        delta_norm=np.concatenate(delta_norms).astype(np.float16),
                        stable=stable.float().cpu().numpy())
    return path


def main() -> None:
    """Command line: ``python -m ablations.export_graphs <run folder> [...]``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    for run_dir in parser.parse_args().runs:
        print(f"{run_dir}: wrote {export(run_dir)}", flush=True)


if __name__ == "__main__":
    main()
