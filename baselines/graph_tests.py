"""Tests of the electrode graphs themselves (ablation_work.md 4.5, electrode-graph level).

Every graph run writes ``graphs.npz``: the electrode graph of every scored
test epoch (``subject``, ``onset``, ``graph [N, V, V]``, and ``channels`` when
the nodes are not our 83 montage electrodes in montage order). The same tests
run on every such file, so a published graph model and our own model are
measured the same way:

- **distance:** Spearman correlation between the mean graph's edge weights and
  the scalp distance of the electrode pair, with the binned decay curve;
- **stage decoding:** how well the sleep stage can be predicted from the graph
  alone (a linear classifier on the graph's edges, leave-one-night-out); a
  graph that is the same for every epoch scores at chance by construction;
- **dynamics:** how much the graph changes from one epoch to the next, and
  around stage transitions compared with elsewhere;
- **similarity:** Spearman correlation between two runs' mean graphs on the
  electrodes they share.

Edges are read as learned channel relationships, not as functional
connectivity. Signed graphs are compared by magnitude where a test needs a
weight (distance, similarity) and used as they are elsewhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
from baselines.common.reference import load_reference_config, scored_epochs

ROOT = Path(__file__).resolve().parent
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def montage(cfg) -> tuple[list[str], np.ndarray]:
    """Our montage electrode names and positions.

    Parameters
    ----------
    cfg : Config
        Reference configuration.

    Returns
    -------
    names : list of str
        Upper-case names in montage order.
    coordinates : np.ndarray
        ``[C, 3]`` positions.
    """
    names, coordinates = data.read_montage(data.find_montage(cfg.paths.data_root))
    return [name.upper() for name in names], coordinates


def load_run(path: Path, names: list[str]) -> dict:
    """One run's graphs, with the montage index of every node (-1 if not an electrode).

    Parameters
    ----------
    path : Path
        ``graphs.npz``.
    names : list of str
        Montage names. Our montage uses the older 10-20 names (T3, T4, T5,
        T6); upstream's preparation writes T7, T8, P7, P8, which
        :data:`ALIASES` maps back.

    Returns
    -------
    dict
        ``subject``, ``onset``, ``graph`` (float32), ``channels``, ``index``.
    """
    stored = np.load(path, allow_pickle=True)
    channels = ([ALIASES.get(str(c).upper(), str(c).upper()) for c in stored["channels"]]
                if "channels" in stored.files else names)
    return {"subject": stored["subject"].astype(str), "onset": stored["onset"].astype(float),
            "graph": stored["graph"].astype(np.float32), "channels": channels,
            "index": np.array([names.index(c) if c in names else -1 for c in channels])}


def stages_of(cfg, run: dict) -> np.ndarray:
    """The scored stage of every graph's epoch.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    run : dict
        Output of :func:`load_run`.

    Returns
    -------
    np.ndarray
        Stage index per graph.

    Raises
    ------
    KeyError
        If a graph's epoch is not a scored test epoch.
    """
    table = scored_epochs(cfg, sorted(set(run["subject"])))
    lookup = {(s, round(float(o), 3)): int(st) for s in table
              for o, st in zip(table[s]["onset"], table[s]["stage"])}
    return np.array([lookup[(s, round(float(o), 3))] for s, o in zip(run["subject"], run["onset"])])


def off_diagonal(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the upper triangle without the diagonal.

    Parameters
    ----------
    size : int
        Nodes.

    Returns
    -------
    tuple of np.ndarray
        Row and column indices.
    """
    return np.triu_indices(size, k=1)


def distance_test(run: dict, coordinates: np.ndarray, bins: int) -> dict:
    """Mean edge magnitude against the scalp distance of the pair.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    coordinates : np.ndarray
        Montage positions.
    bins : int
        Distance quantile bins of the decay curve.

    Returns
    -------
    dict
        ``spearman`` and ``curve`` (bin centre distance, mean magnitude), on
        the nodes that are montage electrodes.
    """
    keep = np.flatnonzero(run["index"] >= 0)
    mean = np.abs(run["graph"].mean(0))[np.ix_(keep, keep)]
    mean = 0.5 * (mean + mean.T)
    position = coordinates[run["index"][keep]]
    distance = np.linalg.norm(position[:, None] - position[None], axis=-1)
    rows, cols = off_diagonal(len(keep))
    weight, dist = mean[rows, cols], distance[rows, cols]
    edges = np.quantile(dist, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, bins - 1)
    curve = [[float(dist[which == b].mean()), float(weight[which == b].mean())]
             for b in range(bins) if np.any(which == b)]
    return {"spearman": float(spearmanr(weight, dist)[0]), "curve": curve, "nodes": int(len(keep))}


def stage_decoding(run: dict, stages: np.ndarray, components: int, seed: int) -> dict:
    """Leave-one-night-out linear decoding of the stage from the graph's edges.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    stages : np.ndarray
        Stage per graph.
    components : int
        Principal components of the edges kept before the classifier.
    seed : int
        Random state of the PCA and the classifier.

    Returns
    -------
    dict
        Pooled ``kappa`` and ``macro_f1`` over the held-out nights, or
        ``constant: true`` when every graph is the same (chance by
        construction); None scores when there is only one night.
    """
    rows, cols = off_diagonal(run["graph"].shape[1])
    edges = run["graph"][:, rows, cols]
    if np.allclose(edges, edges[:1]):
        return {"constant": True, "kappa": 0.0, "macro_f1": None}
    if len(np.unique(run["subject"])) < 2:
        return {"constant": False, "kappa": None, "macro_f1": None,
                "skipped": "leave-one-night-out needs at least two nights"}
    predicted = np.empty_like(stages)
    for subject in np.unique(run["subject"]):
        test = run["subject"] == subject
        model = make_pipeline(StandardScaler(),
                              PCA(min(components, int((~test).sum()) - 1), random_state=seed),
                              LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=seed))
        model.fit(edges[~test], stages[~test])
        predicted[test] = model.predict(edges[test])
    return {"constant": False, "kappa": float(cohen_kappa_score(stages, predicted)),
            "macro_f1": float(f1_score(stages, predicted, average="macro"))}


def dynamics(run: dict, stages: np.ndarray, window: int, epoch_s: float) -> dict:
    """How much the graph changes between consecutive epochs, and around stage transitions.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    stages : np.ndarray
        Stage per graph.
    window : int
        Epochs before and after a transition in the transition-locked curve.
    epoch_s : float
        Seconds per epoch; only epochs exactly this far apart are consecutive.

    Returns
    -------
    dict
        ``change`` (mean relative Frobenius change between consecutive
        epochs), ``transition_ratio`` (mean change at a stage transition over
        the mean change elsewhere) and ``curve`` (mean change at offsets
        ``-window .. window`` from the transitions).
    """
    changes, at_transition, curve = [], [], {k: [] for k in range(-window, window + 1)}
    for subject in np.unique(run["subject"]):
        rows = np.flatnonzero(run["subject"] == subject)
        rows = rows[np.argsort(run["onset"][rows])]
        graph, onset, stage = run["graph"][rows], run["onset"][rows], stages[rows]
        scale = np.linalg.norm(graph, axis=(1, 2)).mean() + 1e-12
        step = np.linalg.norm(np.diff(graph, axis=0), axis=(1, 2)) / scale
        consecutive = np.isclose(np.diff(onset), epoch_s)
        transition = stage[1:] != stage[:-1]
        changes.extend(step[consecutive])
        at_transition.extend(transition[consecutive])
        for position in np.flatnonzero(transition & consecutive):
            for offset in curve:
                k = position + offset
                if 0 <= k < len(step) and consecutive[k]:
                    curve[offset].append(step[k])
    changes, at_transition = np.array(changes), np.array(at_transition, dtype=bool)
    elsewhere = changes[~at_transition].mean() if (~at_transition).any() else np.nan
    return {"change": float(changes.mean()),
            "transition_ratio": float(changes[at_transition].mean() / elsewhere)
            if at_transition.any() and elsewhere > 0 else None,
            "curve": {str(k): float(np.mean(v)) if v else None for k, v in curve.items()}}


def similarity(first: dict, second: dict) -> dict:
    """Spearman correlation of two runs' mean edge magnitudes on their common electrodes.

    Parameters
    ----------
    first, second : dict
        Outputs of :func:`load_run`.

    Returns
    -------
    dict
        ``spearman`` and the number of common electrodes.
    """
    common = sorted(set(first["index"][first["index"] >= 0]) & set(second["index"][second["index"] >= 0]))

    def mean_on(run: dict) -> np.ndarray:
        """The run's mean edge magnitude on the common electrodes, in a shared order.

        Parameters
        ----------
        run : dict
            Output of :func:`load_run`.

        Returns
        -------
        np.ndarray
            Upper-triangle edge magnitudes.
        """
        position = [int(np.flatnonzero(run["index"] == c)[0]) for c in common]
        mean = np.abs(run["graph"].mean(0))[np.ix_(position, position)]
        mean = 0.5 * (mean + mean.T)
        return mean[off_diagonal(len(common))]

    return {"spearman": float(spearmanr(mean_on(first), mean_on(second))[0]),
            "electrodes": len(common)}


def main() -> None:
    """Command line: ``python -m baselines.graph_tests [graphs.npz ...]``.

    Without arguments, every ``graphs.npz`` under the baselines' results
    folder is tested. Settings come from ``baselines/graph_tests.yaml``; the
    results are written to ``<results_dir>/graph_tests.json`` and a table is
    printed.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load((ROOT / "settings.yaml").read_text())
    spec = yaml.safe_load((ROOT / "graph_tests.yaml").read_text())
    results_dir = Path(settings["results_dir"])
    paths = args.paths or sorted(results_dir.rglob("graphs.npz"))
    cfg = load_reference_config(Path(settings["reference_run"]))
    names, coordinates = montage(cfg)
    runs, report = {}, {}
    for path in paths:
        name = str(path.parent.relative_to(results_dir)) if path.is_relative_to(results_dir) \
            else str(path.parent)
        run = load_run(path, names)
        stages = stages_of(cfg, run)
        runs[name] = run
        report[name] = {"epochs": int(len(stages)), "nodes": int(run["graph"].shape[1]),
                        "distance": distance_test(run, coordinates, spec["distance_bins"]),
                        "stage_decoding": stage_decoding(run, stages, spec["pca_components"],
                                                         spec["seed"]),
                        "dynamics": dynamics(run, stages, spec["transition_window"],
                                             cfg.data.window_s)}
    names_list = list(runs)
    report["similarity"] = {f"{a} | {b}": similarity(runs[a], runs[b])
                            for i, a in enumerate(names_list) for b in names_list[i + 1:]}
    (results_dir / "graph_tests.json").write_text(json.dumps(report, indent=1) + "\n")
    def fmt(value: float | None) -> str:
        """Three decimals, or a dash for a missing value.

        Parameters
        ----------
        value : float or None
            Score.

        Returns
        -------
        str
            Formatted value.
        """
        return "—" if value is None else f"{value:.3f}"

    print("| Run | Nodes | Epochs | ρ(weight, distance) | stage from graph κ | change/epoch | "
          "change at transitions ÷ elsewhere |")
    print("|---|---|---|---|---|---|---|")
    for name in names_list:
        r = report[name]
        ratio = r["dynamics"]["transition_ratio"]
        print(f"| {name} | {r['nodes']} | {r['epochs']} | {r['distance']['spearman']:+.3f} "
              f"| {fmt(r['stage_decoding']['kappa'])} | {r['dynamics']['change']:.3f} "
              f"| {'—' if ratio is None else f'{ratio:.2f}'} |")


if __name__ == "__main__":
    main()
