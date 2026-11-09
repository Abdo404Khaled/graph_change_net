from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
REFERENCE = "exp15_final"
COLUMNS = ("kappa", "macro_f1", "accuracy", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_R")


def fold_of(path: Path) -> int:
    """The fold index in a run folder name ``<name>_fold<k>``.

    Parameters
    ----------
    path : Path
        A run's ``metrics.json``.

    Returns
    -------
    int
        ``k``.
    """
    return int(path.parent.name.rsplit("_fold", 1)[1])


def runs(folder: Path, name: str) -> dict[int, dict]:
    """The finished folds of one run name.

    Parameters
    ----------
    folder : Path
        Folder holding ``<name>_fold<k>/metrics.json``.
    name : str
        Run name without the fold suffix.

    Returns
    -------
    dict
        ``fold -> metrics``; quick runs (``_quick_fold``) are not included.
    """
    return {fold_of(path): json.loads(path.read_text())
            for path in folder.glob(f"{name}_fold*/metrics.json")}


def reconstruction_share(metrics: dict) -> float:
    """Variance of masked whole electrodes explained by the model, minus nearest-electrode's.

    Parameters
    ----------
    metrics : dict
        A run's ``metrics.json``.

    Returns
    -------
    float
        Positive when the model reconstructs hidden electrodes better than
        copying the nearest visible one; NaN without a reconstruction report.
    """
    rows = {(row["regime"], row["predictor"]): row["variance_explained"]
            for row in metrics.get("reconstruction") or []}
    model = rows.get(("whole_electrodes", "model"))
    nearest = rows.get(("whole_electrodes", "nearest_electrode"))
    return float("nan") if model is None or nearest is None else model - nearest


def table() -> str:
    """Every ablation against Exp 15, fold by fold, kappa and macro-F1 first.

    Each row gives the scores on the folds that have finished, the mean over
    them, and the paired difference to Exp 15 on the same folds, with the
    number of those folds on which the ablation is lower. Also: the
    parameter count, the kappa lost when the 30 segments are shuffled (the
    temporal-dynamics control) and the reconstruction of masked electrodes
    relative to the nearest-electrode baseline.

    Returns
    -------
    str
        Markdown table.
    """
    reference = runs(REPOSITORY / "runs", REFERENCE)
    header = ("| Run | Params | Folds | κ | macro-F1 | Δκ vs Exp 15 | Δ macro-F1 | lower on |"
              " shuffled Δκ | recon. vs nearest |")
    lines = [header, "|" + "---|" * 10]
    names = [REFERENCE] + sorted(path.stem for path in (ROOT / "configs").glob("*.yaml"))
    for name in names:
        found = reference if name == REFERENCE else runs(ROOT / "results", name)
        if not found:
            lines.append(f"| {name} | | 0 | | | | | | | |")
            continue
        folds = sorted(found)
        kappa = np.array([found[k]["test"]["kappa"] for k in folds])
        macro = np.array([found[k]["test"]["macro_f1"] for k in folds])
        paired = [k for k in folds if k in reference]
        delta_kappa = np.array([found[k]["test"]["kappa"] - reference[k]["test"]["kappa"]
                                for k in paired])
        delta_macro = np.array([found[k]["test"]["macro_f1"] - reference[k]["test"]["macro_f1"]
                                for k in paired])
        shuffled = np.mean([found[k]["test_shuffled"]["delta_kappa"]
                            for k in folds if "test_shuffled" in found[k]])
        recon = np.nanmean([reconstruction_share(found[k]) for k in folds])
        spread = f" ± {kappa.std():.4f}" if len(folds) > 1 else ""
        lines.append(
            f"| {name} | {found[folds[0]]['parameters']:,} | {','.join(map(str, folds))} "
            f"| {kappa.mean():.4f}{spread} | {macro.mean():.4f} "
            f"| {delta_kappa.mean():+.4f} | {delta_macro.mean():+.4f} "
            f"| {int((delta_kappa < 0).sum())}/{len(paired)} | {shuffled:+.4f} | {recon:+.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
