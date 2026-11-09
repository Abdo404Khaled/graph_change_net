"""Left-right against front-back orientation of the stage-specific edges.

For each sleep stage, the deviation of the stage mean of ``A_t`` from the mean over all test epochs
is computed per electrode pair (as in ``stage_figure.py``, Fig. 10). Each edge is projected onto the
left-right axis (|Δy|) and the front-back axis (|Δx|) of the montage, both in the head frame (x
forward, y left), and the projections are accumulated with the edge's |deviation| as weight:

    LR = Σ |dev_e| |Δy_e|,   FB = Σ |dev_e| |Δx_e|,   orientation = (LR / FB) / (LR_all / FB_all)

where ``LR_all / FB_all`` is the same ratio over all electrode pairs with equal weight, so an
orientation of 1 means no preference beyond the montage geometry and values above 1 mean a shift
toward left-right edges. Computed for the 24 notable edges of Fig. 10 (12 stronger, 12 weaker, among
edges significant at p = 1/201), for every significant edge, and separately for strengthened and
weakened edges; for fold 0 with a within-night permutation null (stage labels shuffled within each
night, 200 draws, the selection repeated on every draw) and for folds 0–4 (range over folds).
argv: output JSON path.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from baselines.common.reference import load_reference_config  # noqa: E402
from baselines.graph_tests import load_run, montage, stages_of  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "R")
PERMUTATIONS = 200
NOTABLE = 12


def deviations(values: np.ndarray, stage: np.ndarray) -> np.ndarray:
    """Stage mean minus overall mean for every feature.

    Parameters
    ----------
    values : np.ndarray
        ``[N, E]`` per-epoch edge weights.
    stage : np.ndarray
        Stage index per epoch.

    Returns
    -------
    np.ndarray
        ``[5, E]`` deviations.
    """
    mean = values.mean(0)
    return np.array([values[stage == k].mean(0) - mean for k in range(5)])


def shuffled(stage: np.ndarray, subject: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stage labels permuted within each night.

    Parameters
    ----------
    stage : np.ndarray
        Stage index per epoch.
    subject : np.ndarray
        Night per epoch.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    np.ndarray
        Permuted labels.
    """
    out = stage.copy()
    for night in np.unique(subject):
        index = np.flatnonzero(subject == night)
        out[index] = rng.permutation(stage[index])
    return out


def orientation(weights: np.ndarray, lateral: np.ndarray, frontal: np.ndarray,
                baseline: float) -> dict:
    """Weighted left-right and front-back projections of a set of edges.

    Parameters
    ----------
    weights : np.ndarray
        |deviation| of each selected edge.
    lateral, frontal : np.ndarray
        |Δy| and |Δx| of each selected edge.
    baseline : float
        ``LR_all / FB_all`` over all pairs with equal weight.

    Returns
    -------
    dict
        ``left_right`` and ``front_back`` accumulated projections, ``share_left_right``
        (LR / (LR + FB)) and ``orientation`` (geometry-normalised ratio).
    """
    if weights.sum() == 0:
        return {"left_right": 0.0, "front_back": 0.0, "share_left_right": float("nan"),
                "orientation": float("nan")}
    lr, fb = float((weights * lateral).sum()), float((weights * frontal).sum())
    return {"left_right": lr, "front_back": fb, "share_left_right": lr / (lr + fb),
            "orientation": (lr / fb) / baseline}


def stage_orientations(dev: np.ndarray, significant: np.ndarray, lateral: np.ndarray,
                       frontal: np.ndarray, baseline: float) -> dict:
    """Orientation of the notable, all-significant, strengthened and weakened edges of every stage.

    Parameters
    ----------
    dev : np.ndarray
        ``[5, E]`` deviations.
    significant : np.ndarray
        ``[5, E]`` boolean, significant edges.
    lateral, frontal : np.ndarray
        ``[E]`` |Δy| and |Δx|.
    baseline : float
        Geometry ratio.

    Returns
    -------
    dict
        Stage name to the four orientation summaries.
    """
    result = {}
    for k, name in enumerate(STAGES):
        index = np.flatnonzero(significant[k])
        stronger = index[dev[k, index] > 0]
        weaker = index[dev[k, index] < 0]
        notable = np.concatenate([stronger[np.argsort(-dev[k, stronger])][:NOTABLE],
                                  weaker[np.argsort(dev[k, weaker])][:NOTABLE]])
        entry = {}
        for label, chosen in (("notable", notable), ("significant", index),
                              ("stronger", stronger), ("weaker", weaker)):
            entry[label] = orientation(np.abs(dev[k, chosen]), lateral[chosen], frontal[chosen],
                                       baseline)
            entry[label]["edges"] = int(len(chosen))
        result[name] = entry
    return result


def main():
    """Fold 0 with the permutation null, folds 0–4 for the range, written to one JSON."""
    cfg = load_reference_config(REPO / "runs/exp15_final_fold0")
    names, coords = montage(cfg)
    upper = np.triu_indices(len(names), 1)
    lateral = np.abs(coords[upper[0], 1] - coords[upper[1], 1])
    frontal = np.abs(coords[upper[0], 0] - coords[upper[1], 0])
    baseline = float(lateral.sum() / frontal.sum())
    rng = np.random.default_rng(0)
    folds = {}
    for fold in range(5):
        path = REPO / f"runs/exp15_final_fold{fold}/graphs.npz"
        run = load_run(path, names)
        stage, subject = stages_of(cfg, run), run["subject"]
        edges = run["graph"][:, upper[0], upper[1]].astype(np.float32)
        observed = deviations(edges, stage)
        exceed = np.zeros_like(observed)
        nulls = []
        for _ in range(PERMUTATIONS):
            permuted = deviations(edges, shuffled(stage, subject, rng))
            exceed += np.abs(permuted) >= np.abs(observed)
            if fold == 0:
                nulls.append(permuted)
        significant = (1 + exceed) / (PERMUTATIONS + 1) <= 1 / (PERMUTATIONS + 1)
        folds[fold] = stage_orientations(observed, significant, lateral, frontal, baseline)
        if fold == 0:
            null_summary = {name: {"notable": [], "significant_top": [], "stronger": [], "weaker": []} for name in STAGES}
            for permuted in nulls:
                top = np.zeros_like(permuted, dtype=bool)
                for k in range(5):
                    order = np.argsort(-np.abs(permuted[k]))[:int(significant[k].sum()) or 1]
                    top[k, order] = True
                entry = stage_orientations(permuted, top, lateral, frontal, baseline)
                for name in STAGES:
                    null_summary[name]["notable"].append(entry[name]["notable"]["orientation"])
                    null_summary[name]["significant_top"].append(entry[name]["significant"]["orientation"])
                    null_summary[name]["stronger"].append(entry[name]["stronger"]["orientation"])
                    null_summary[name]["weaker"].append(entry[name]["weaker"]["orientation"])
        print(fold, {name: round(folds[fold][name]["significant"]["orientation"], 3)
                     for name in STAGES}, flush=True)
    result = {"baseline_lr_over_fb_all_pairs": baseline, "permutations": PERMUTATIONS,
              "notable_per_sign": NOTABLE, "fold0": folds[0], "folds": {}, "null_fold0": {}}
    for name in STAGES:
        for label in ("notable", "significant", "stronger", "weaker"):
            values = [folds[f][name][label]["orientation"] for f in range(5)]
            result["folds"].setdefault(name, {})[label] = {"min": float(np.nanmin(values)),
                                                           "max": float(np.nanmax(values)),
                                                           "mean": float(np.nanmean(values))}
        for label, key in (("notable", "notable"), ("significant", "significant_top"),
                           ("stronger", "stronger"), ("weaker", "weaker")):
            null = np.array(null_summary[name][key])
            value = folds[0][name][label]["orientation"]
            result["null_fold0"].setdefault(name, {})[label] = {
                "null_mean": float(np.nanmean(null)), "null_2.5": float(np.nanpercentile(null, 2.5)),
                "null_97.5": float(np.nanpercentile(null, 97.5)),
                "p_two_sided": float(min(1.0, 2 * min((1 + (null >= value).sum()) / (len(null) + 1),
                                                      (1 + (null <= value).sum()) / (len(null) + 1))))}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
    for name in STAGES:
        s = folds[0][name]
        print(f"{name}: significant {s['significant']['edges']} edges, orientation "
              f"{s['significant']['orientation']:.3f} (folds {result['folds'][name]['significant']['min']:.3f}–"
              f"{result['folds'][name]['significant']['max']:.3f}; null {result['null_fold0'][name]['significant']['null_2.5']:.3f}–"
              f"{result['null_fold0'][name]['significant']['null_97.5']:.3f}, p {result['null_fold0'][name]['significant']['p_two_sided']:.3f}) | "
              f"notable {s['notable']['orientation']:.3f} | stronger {s['stronger']['orientation']:.3f} | weaker {s['weaker']['orientation']:.3f}")


if __name__ == "__main__":
    main()
