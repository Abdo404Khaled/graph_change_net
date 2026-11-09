"""Spatial structure of the learned stable graph L^S (Section V-C of the paper).

For every run with an electrode-graph learner, ``L^S`` is rebuilt from the run's
``classify_best.pt`` and compared with the montage geometry: signed and absolute Spearman
correlation with closeness (negative distance), the share of each electrode's strongest positive edges that go to its nearest
electrodes, the mean length of the strongest edges, the share of negative entries, how often the
strongest positive partner lies in the far half of the montage. argv: output JSON path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from config import load_run_config  # noqa: E402

RUNS = {f"Exp 15 fold {k}": REPO / f"runs/exp15_final_fold{k}" for k in range(5)}
RUNS |= {"15c fold 0": REPO / "runs/exp15c_reconstruction_only_fold0",
         "15b fold 0": REPO / "runs/exp15b_no_context_fold0",
         "15a (no stage 1) fold 0": REPO / "runs/exp15a_no_stage1_fold0",
         "final12 fold 0": REPO / "runs/final12_window_graph_fold0",
         "final5 fold 0": REPO / "runs/final5_signed_gcn_fold0"}
RUNS |= {name: REPO / f"ablations/results/{name}_fold0" for name in
         ("stable_only", "no_memory", "epoch_graph", "no_segment_graph", "cluster_mask", "wideband")}


def stable_graph(run_dir: Path) -> np.ndarray | None:
    """``L^S`` of a run from its stage-2 weights, or None if the run has no embedding learner.

    Parameters
    ----------
    run_dir : Path
        Run folder.

    Returns
    -------
    np.ndarray or None
        ``[C, C]`` symmetric stable graph.
    """
    state = torch.load(run_dir / "classify_best.pt", map_location="cpu")
    if "graph_learner.embed_from" not in state:
        return None
    left = state["graph_learner.embed_from"].numpy()
    right = state["graph_learner.embed_to"].numpy()
    scores = left @ right.T / np.sqrt(left.shape[1])
    return 0.5 * (scores + scores.T)


def structure(graph: np.ndarray, distance: np.ndarray) -> dict:
    """Geometry measures of one stable graph.

    Parameters
    ----------
    graph : np.ndarray
        ``[C, C]`` stable graph.
    distance : np.ndarray
        ``[C, C]`` electrode distances.

    Returns
    -------
    dict
        The measures listed in the module docstring.
    """
    n = len(graph)
    upper = np.triu_indices(n, 1)
    off = ~np.eye(n, dtype=bool)
    mean_distance = distance[upper].mean()
    nearest = np.argsort(np.where(off, distance, np.inf), 1)
    positive = np.where(off, graph, -np.inf)
    negative = np.where(off, -graph, -np.inf)
    top3_pos, top3_neg = np.argsort(-positive, 1)[:, :3], np.argsort(-negative, 1)[:, :3]
    top5_pos = np.argsort(-positive, 1)[:, :5]
    rank = np.argsort(np.argsort(np.where(off, distance, -np.inf), 1), 1)
    rows = np.arange(n)[:, None]
    def overlap(top: np.ndarray, k: int) -> float:
        """Mean share of each electrode's ``top`` partners among its ``k`` nearest electrodes."""
        return float(np.mean([len(set(top[i]) & set(nearest[i, :k])) / k for i in range(n)]))

    far = rank[np.arange(n), top3_pos[:, 0]] > (n - 1) / 2
    return {"signed_rho": float(spearmanr(graph[upper], -distance[upper]).correlation),
            "abs_rho": float(spearmanr(np.abs(graph[upper]), -distance[upper]).correlation),
            "top3_pos_in_3_nearest": overlap(top3_pos, 3),
            "mean_dist_top3_pos": float(distance[rows, top3_pos].mean() / mean_distance),
            "mean_dist_top3_neg": float(distance[rows, top3_neg].mean() / mean_distance),
            "negative_share": float((graph[off] < 0).mean()),
            "strongest_positive_partner_is_far": float(far.mean()),
            "top5_positive_within_5_nearest": overlap(top5_pos, 5)}


def main():
    """Measure every available run and write the JSON."""
    cfg = load_run_config(RUNS["Exp 15 fold 0"], {})
    _, coords = data.read_montage(data.find_montage(cfg.paths.data_root))
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    result = {}
    for name, run_dir in RUNS.items():
        if not (run_dir / "classify_best.pt").exists():
            continue
        graph = stable_graph(run_dir)
        if graph is not None:
            result[name] = structure(graph, distance)
            print(name, {k: round(v, 3) for k, v in result[name].items()}, flush=True)
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
