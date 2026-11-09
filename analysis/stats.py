"""Paired per-night bootstrap of Exp 15 against every fold-0 model, with Holm correction per family.

Every prediction set is aligned to the same 4,462 scored test epochs (baselines.common.scoring.align).
One bootstrap draw resamples the 5 test nights with replacement and then the epochs within each drawn
night with replacement; the same draw scores Exp 15 and the comparator, so the difference is paired.
Native AnySleep and U-Sleep, and causal AnySleep, are scored as the mean over their 3 released runs, as
reported. The baselines family also holds the causal and CNN-2S configurations of the single split, and the
components family the run trained without edge signs (``unsigned_graph``, from ``t1_predictions.py``).
p is two-sided (2 x the smaller tail of the bootstrap difference). Also writes per-night kappa.
argv: folder with e2_all predictions, output folder.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from baselines.common.reference import load_reference_config, scored_epochs  # noqa: E402
from baselines.common.scoring import align  # noqa: E402

PRED = Path(sys.argv[1]) / "predictions"
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
B = 5000
cfg = load_reference_config(REPO / "runs/exp15_final_fold0")
test = json.loads((REPO / "baselines/folds/fold0.json").read_text())["test"]
table = scored_epochs(cfg, test)
R = REPO / "baselines/results"
families = {
    "baselines": {
        **{f"{m} (native)": [R / m / "native/predictions.npz"] for m in
           ["cbramod", "deepsleepnet", "hypnopc", "graphsleepnet", "graphsleepnet_distance", "mstgcn", "productgraphsleepnet"]},
        "anysleep (native)": [R / f"anysleep/native/anysleep-run{k}/predictions.npz" for k in (1, 2, 3)],
        "usleep (native)": [R / f"usleep/native/usleep-run{k}_soft/predictions.npz" for k in (1, 2, 3)],
        **{f"{m} (ours)": [R / m / "ours/predictions.npz"] for m in
           ["anysleep", "usleep", "deepsleepnet", "cbramod", "hypnopc", "graphsleepnet", "graphsleepnet_distance",
            "mstgcn", "productgraphsleepnet"]},
        **{f"{m} (causal)": [R / m / "causal/predictions.npz"] for m in
           ["graphsleepnet", "mstgcn", "productgraphsleepnet"]},
        "anysleep (causal)": [R / f"anysleep/causal/anysleep-run{k}/predictions.npz" for k in (1, 2, 3)],
        **{f"{m} (cnn2s)": [R / m / "cnn2s/predictions.npz"] for m in ["graphsleepnet", "productgraphsleepnet"]},
    },
    "components": {n: [PRED / f"{n.replace(' ', '_')}.npz"] for n in
                   ["15a no stage 1", "15b no context", "15c reconstruction only", "distance_graph", "stable_only",
                    "no_memory", "no_segment_graph", "epoch_graph", "no_electrode_graph", "no_graph", "unsigned_graph"]},
    "settings": {n: [PRED / f"{n}.npz"] for n in ["cluster_mask", "wideband", "wideband_cluster", "raw", "electrodes_19"]},
}
extra = {"no_electrode_graph_no_stage1 vs 15a": ([PRED / "no_electrode_graph_no_stage1.npz"], [PRED / "15a_no_stage_1.npz"])}


def load(paths):
    out = []
    for path in paths:
        true, predicted, _, subject = align(table, dict(np.load(path, allow_pickle=True)))
        out.append(predicted)
    return true, out, subject


def kappa_f1(true, predicted, index):
    t, p = true[index], predicted[index]
    cm = np.bincount(t * 5 + p, minlength=25).reshape(5, 5).astype(float)
    n = cm.sum(); po = np.trace(cm) / n; pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    kappa = (po - pe) / (1 - pe)
    tp = np.diag(cm); precision = tp / np.maximum(cm.sum(0), 1); recall = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    return kappa, f1.mean()


reference_true, (reference,), subject = load([PRED / "Exp_15.npz"])
nights = {s: np.flatnonzero(subject == s) for s in test}
rng = np.random.default_rng(0)
draws = []
for _ in range(B):
    chosen = rng.choice(test, len(test), replace=True)
    draws.append(np.concatenate([rng.choice(nights[s], len(nights[s]), replace=True) for s in chosen]))
full = np.arange(len(reference_true))
base = np.array([kappa_f1(reference_true, reference, d) for d in draws])


def compare(paths, against=None):
    true, predictions, _ = load(paths)
    assert np.array_equal(true, reference_true)
    ref_predictions = [reference] if against is None else load(against)[1]
    point = np.mean([kappa_f1(true, p, full) for p in predictions], axis=0)
    ref_point = np.mean([kappa_f1(true, p, full) for p in ref_predictions], axis=0)
    ref_boot = base if against is None else np.mean([[kappa_f1(true, p, d) for d in draws] for p in ref_predictions], axis=0)
    boot = np.mean([[kappa_f1(true, p, d) for d in draws] for p in predictions], axis=0)
    diff = ref_boot - boot
    result = {"kappa": float(point[0]), "macro_f1": float(point[1]),
              "delta_kappa": float(ref_point[0] - point[0]), "delta_macro_f1": float(ref_point[1] - point[1])}
    for i, key in enumerate(["kappa", "macro_f1"]):
        result[f"ci_delta_{key}"] = [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]
        result[f"p_{key}"] = float(min(1.0, 2 * min((diff[:, i] <= 0).mean(), (diff[:, i] >= 0).mean())))
    per_night = {s: float(kappa_f1(true, predictions[0], nights[s])[0]) for s in test}
    return result, per_night


def holm(results, key):
    order = sorted(results, key=lambda n: results[n][key])
    m, running = len(order), 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * results[name][key]))
        results[name][f"{key}_holm"] = running


output = {"B": B, "exp15": {"kappa": float(kappa_f1(reference_true, reference, full)[0]),
                            "macro_f1": float(kappa_f1(reference_true, reference, full)[1])},
          "per_night": {"Exp 15": {s: float(kappa_f1(reference_true, reference, nights[s])[0]) for s in test}}}
for family, members in families.items():
    output[family] = {}
    for name, paths in members.items():
        if not all(p.exists() for p in paths):
            print("missing", name, flush=True)
            continue
        output[family][name], output["per_night"][name] = compare(paths)
        r = output[family][name]
        print(family, name, f"k {r['kappa']:.4f} dk {r['delta_kappa']:+.4f} CI [{r['ci_delta_kappa'][0]:+.4f},{r['ci_delta_kappa'][1]:+.4f}] p {r['p_kappa']:.4f}", flush=True)
    holm(output[family], "p_kappa"); holm(output[family], "p_macro_f1")
output["extra"] = {}
for name, (paths, against) in extra.items():
    if all(p.exists() for p in paths + against):
        output["extra"][name], _ = compare(paths, against)
        print("extra", name, output["extra"][name], flush=True)
(OUT / "stats.json").write_text(json.dumps(output, indent=1))
print("done", flush=True)
