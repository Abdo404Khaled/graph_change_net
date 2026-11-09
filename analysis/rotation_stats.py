"""Per-night and pooled comparison of Exp 15 with every baseline over the rotation's test nights.

Exp 15 predictions come from rotation_outputs.py (one file per rotation group); baselines from
``baselines/results/<model>/<config>_rot<k>/predictions.npz`` (AnySleep and U-Sleep: their 3
released runs, scored separately and averaged, as in the main table). Only rotation groups that
both Exp 15 and the baseline have finished are used, and every prediction set is aligned to the same
scored epochs of the group's test nights. For each comparison: pooled kappa and macro-F1 of both
models over those epochs, per-night kappa, and a paired bootstrap over nights (resample nights with
replacement, then epochs within each drawn night), B = 5000, two-sided p; Holm over the family, for kappa and for macro-F1.
argv: folder of rotation predictions, output JSON path.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from baselines.common.reference import load_reference_config, scored_epochs  # noqa: E402
from baselines.common.scoring import align  # noqa: E402

PRED = Path(sys.argv[1])
OUT = Path(sys.argv[2])
R = REPO / "baselines/results"
B = 5000
BASELINES = {
    "AnySleep (native, zero-shot)": ("anysleep", "native", True),
    "U-Sleep (native, zero-shot)": ("usleep", "native", True),
    "Hypno-PC (native)": ("hypnopc", "native", False),
    "GraphSleepNet (native)": ("graphsleepnet", "native", False),
    "GraphSleepNet-dist. (native)": ("graphsleepnet_distance", "native", False),
    "MSTGCN (native)": ("mstgcn", "native", False),
    "ProductGraphSleepNet (native)": ("productgraphsleepnet", "native", False),
    "GraphSleepNet (CNN-2S)": ("graphsleepnet", "cnn2s", False),
    "ProductGraphSleepNet (CNN-2S)": ("productgraphsleepnet", "cnn2s", False),
    "AnySleep (causal, zero-shot)": ("anysleep", "causal", True),
    "GraphSleepNet (causal)": ("graphsleepnet", "causal", False),
    "MSTGCN (causal)": ("mstgcn", "causal", False),
    "ProductGraphSleepNet (causal)": ("productgraphsleepnet", "causal", False),
}
cfg = load_reference_config(REPO / "runs/exp15_final_fold0")


def scores(true, predicted, index):
    """Kappa and macro-F1 of the epochs in ``index``."""
    t, p = true[index], predicted[index]
    cm = np.bincount(t * 5 + p, minlength=25).reshape(5, 5).astype(float)
    n = cm.sum(); po = np.trace(cm) / n; pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    tp = np.diag(cm); pr = tp / np.maximum(cm.sum(0), 1); rc = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0)
    return (po - pe) / (1 - pe), f1.mean()


def baseline_files(model, config, runs, group):
    """Prediction files of one baseline for one rotation group, or None if not finished."""
    folder = R / model / f"{config}_rot{group}"
    files = sorted(p for p in folder.glob("*-run*/predictions.npz") if "ensemble" not in str(p)) \
        if runs else [folder / "predictions.npz"]
    if model == "usleep":
        files = sorted(folder.glob("usleep-run*_soft/predictions.npz")) or files
    return files if files and all(f.exists() for f in files) else None


def main():
    """Compare Exp 15 with every baseline on the rotation groups both have finished."""
    groups = sorted(int(p.stem[3:]) for p in PRED.glob("rot*.npz"))
    tables = {}
    for g in groups:
        test = json.loads((REPO / f"baselines/folds/rot{g}.json").read_text())["test"]
        tables[g] = scored_epochs(cfg, test)
    ours = {g: align(tables[g], dict(np.load(PRED / f"rot{g}.npz", allow_pickle=True))) for g in groups}
    rng = np.random.default_rng(0)
    results = {"exp15_groups": groups}
    for name, (model, config, runs) in BASELINES.items():
        used = [g for g in groups if baseline_files(model, config, runs, g)]
        if not used:
            continue
        true = np.concatenate([ours[g][0] for g in used])
        mine = np.concatenate([ours[g][1] for g in used])
        subject = np.concatenate([ours[g][3] for g in used])
        theirs = []
        for path_list in zip(*[baseline_files(model, config, runs, g) for g in used]):
            parts = [align(tables[g], dict(np.load(p, allow_pickle=True))) for g, p in zip(used, path_list)]
            assert np.array_equal(np.concatenate([q[0] for q in parts]), true)
            theirs.append(np.concatenate([q[1] for q in parts]))
        nights = {s: np.flatnonzero(subject == s) for s in np.unique(subject)}
        full = np.arange(len(true))
        k0, f0 = scores(true, mine, full)
        kb = np.mean([scores(true, p, full)[0] for p in theirs]); fb = np.mean([scores(true, p, full)[1] for p in theirs])
        per_night = {s: {"exp15": scores(true, mine, i)[0], "baseline": float(np.mean([scores(true, p, i)[0] for p in theirs]))}
                     for s, i in nights.items()}
        names = list(nights)
        diffs_k, diffs_f = [], []
        for _ in range(B):
            chosen = rng.choice(names, len(names), replace=True)
            d = np.concatenate([rng.choice(nights[s], len(nights[s]), replace=True) for s in chosen])
            a = scores(true, mine, d)
            b = [scores(true, p, d) for p in theirs]
            diffs_k.append(a[0] - np.mean([x[0] for x in b])); diffs_f.append(a[1] - np.mean([x[1] for x in b]))
        diffs_k, diffs_f = np.array(diffs_k), np.array(diffs_f)
        p_k = min(1.0, 2 * min((diffs_k <= 0).mean(), (diffs_k >= 0).mean()))
        p_f = min(1.0, 2 * min((diffs_f <= 0).mean(), (diffs_f >= 0).mean()))
        wins = sum(v["exp15"] > v["baseline"] for v in per_night.values())
        results[name] = {"groups": used, "nights": len(names), "epochs": int(len(true)),
                         "exp15_kappa": k0, "baseline_kappa": float(kb), "delta_kappa": k0 - kb,
                         "ci_kappa": [float(np.percentile(diffs_k, 2.5)), float(np.percentile(diffs_k, 97.5))], "p_kappa": p_k,
                         "exp15_macro_f1": f0, "baseline_macro_f1": float(fb), "delta_macro_f1": f0 - fb,
                         "ci_macro_f1": [float(np.percentile(diffs_f, 2.5)), float(np.percentile(diffs_f, 97.5))], "p_macro_f1": p_f,
                         "nights_exp15_ahead": int(wins), "per_night": per_night}
        print(f"{name:<32} groups {used} nights {len(names):>2}: kappa {k0:.4f} vs {kb:.4f} "
              f"(d {k0-kb:+.4f}, CI [{results[name]['ci_kappa'][0]:+.3f}, {results[name]['ci_kappa'][1]:+.3f}], p {p_k:.3f}) | "
              f"MF1 {f0:.4f} vs {fb:.4f} (d {f0-fb:+.4f}, p {p_f:.3f}) | ahead on {wins}/{len(names)} nights", flush=True)
    names = [n for n in results if n != "exp15_groups"]
    for metric in ("kappa", "macro_f1"):
        order = sorted(names, key=lambda n: results[n][f"p_{metric}"])
        m = len(order)
        running = 0.0
        for i, n in enumerate(order):
            running = max(running, min(1.0, (m - i) * results[n][f"p_{metric}"]))
            results[n][f"holm_p_{metric}"] = running
    OUT.write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
