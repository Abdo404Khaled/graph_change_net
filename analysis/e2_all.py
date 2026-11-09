"""Full-test predictions of every fold-0 model on the Exp 15 cache, and E2 (electrodes removed) for all.

One pass over the 4,462 scored test epochs feeds every model (IO is the bottleneck): Exp 15,
15a/15b/15c, the hierarchy ablations, cluster_mask fold 0, and the 8 baselines trained under our
configuration (built by their runners, loaded from classify_best.pt). Predictions are saved for the
bootstrap and scored with baselines.common.scoring.score_run (checked against each run's reported
kappa). E2 uses the stratified subset (up to 400 epochs per stage)
and the same electrode draws (rng seed 0), electrodes zeroed after scaling, for 10/25/50/75 %.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from baselines.common.ours import OurProtocolModel, item_keys  # noqa: E402
from baselines.common.reference import scored_epochs  # noqa: E402
from baselines.common.scoring import save_predictions, score_run  # noqa: E402
from config import load_run_config  # noqa: E402
from evaluate import scores  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

OUT = Path(sys.argv[1]); (OUT / "predictions").mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "ablations/results"
OURS = {"Exp 15": REPO / "runs/exp15_final_fold0", "15a no stage 1": REPO / "runs/exp15a_no_stage1_fold0",
        "15b no context": REPO / "runs/exp15b_no_context_fold0",
        "15c reconstruction only": REPO / "runs/exp15c_reconstruction_only_fold0"}
OURS |= {n: RESULTS / f"{n}_fold0" for n in ["distance_graph", "stable_only", "no_memory", "no_segment_graph",
                                             "epoch_graph", "no_electrode_graph", "no_graph",
                                             "no_electrode_graph_no_stage1", "cluster_mask"]}
BASELINES = ["anysleep", "usleep", "deepsleepnet", "cbramod", "graphsleepnet",
             "graphsleepnet_distance", "mstgcn", "productgraphsleepnet"]
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75]

cfg = load_run_config(OURS["Exp 15"], {})
split = json.loads((OURS["Exp 15"] / "metrics.json").read_text())["split"]
arrays, channels, coords = data.load_dataset(cfg)
loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)
test = loaders["test"]
np.savez(OUT / "scaler_fold0.npz", center=test.dataset.center[:, 0], scale=test.dataset.scale[:, 0])
names, onsets = item_keys(test, split["test"])
table = scored_epochs(cfg, split["test"])
device = cfg.device

models, reported = {}, {}
for name, run_dir in OURS.items():
    run_cfg = load_run_config(run_dir, {})
    model = GraphChangeNet(run_cfg, len(channels), coords).to(device)
    load_weights(model, run_dir / "classify_best.pt", device)
    models[name] = model.eval()
    reported[name] = json.loads((run_dir / "metrics.json").read_text())["test"]["kappa"]

settings = yaml.safe_load((REPO / "baselines/settings.yaml").read_text())
settings = {**settings, "reference_run": str(REPO / settings["reference_run"]), "fold": str(REPO / settings["fold"])}


class Captured(Exception):
    pass


def capture(network, *args, **kwargs):
    raise Captured(network)


for name in BASELINES:
    runner = importlib.import_module(f"baselines.{name}.runner")
    for attribute in ("run_ours", "run_graph_ours"):
        if hasattr(runner, attribute):
            setattr(runner, attribute, capture)
    spec = yaml.safe_load((REPO / f"baselines/{name}/ours.yaml").read_text())
    try:
        runner.run(spec, settings, OUT / "unused" / name, False)
    except Captured as caught:
        network = caught.args[0]
    wrapped = OurProtocolModel(network)
    wrapped.load_state_dict(torch.load(REPO / f"baselines/results/{name}/ours/classify_best.pt", map_location="cpu"))
    models[f"{name} (ours)"] = wrapped.to(device).eval()
    reported[f"{name} (ours)"] = json.loads((REPO / f"baselines/results/{name}/ours/metrics.json").read_text())["test"]["kappa"]
print("models", len(models), flush=True)


@torch.no_grad()
def run_all(loader, transform=None):
    true, probability = [], {name: [] for name in models}
    for x, y in loader:
        x = x.to(device)
        if transform is not None:
            x = transform(x)
        keep = (y.flatten() >= 0).numpy()
        true.append(y.flatten().numpy()[keep])
        for name, model in models.items():
            logits = model(x)[0]
            probability[name].append(logits.flatten(0, -2).float().softmax(-1).cpu().numpy()[keep])
    return np.concatenate(true), {name: np.concatenate(p) for name, p in probability.items()}


results = {"full_test": {}, "E2": {name: {} for name in models}}
true, probability = run_all(test)
for name, p in probability.items():
    path = OUT / "predictions" / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
    save_predictions(path, np.array(names), np.array(onsets), p.argmax(1), p)
    result = score_run(cfg, table, path)
    results["full_test"][name] = {**{k: result["test"][k] for k in ("kappa", "macro_f1", "accuracy")},
                                  "reported_kappa": reported[name]}
    print("full", name, round(result["test"]["kappa"], 4), "reported", round(reported[name], 4), flush=True)
(OUT / "e2_all.json").write_text(json.dumps(results, indent=1))

rng = np.random.default_rng(0)
subset = np.sort(np.concatenate([rng.choice(np.flatnonzero(true == k), min(400, (true == k).sum()), replace=False)
                                 for k in range(5)]))
draws = {f: torch.as_tensor(rng.permutation(len(channels))[:int(round(f * len(channels)))], device=device)
         for f in FRACTIONS}
results["subset_size"] = int(len(subset))
results["removed_electrodes"] = {str(f): [channels[i] for i in draws[f].tolist()] for f in FRACTIONS}
sub_loader = DataLoader(Subset(test.dataset, subset), batch_size=8, num_workers=8)


def drop(index):
    def transform(x):
        x = x.clone()
        x[:, :, index] = 0
        return x
    return transform


for f in FRACTIONS:
    sub_true, sub_probability = run_all(sub_loader, drop(draws[f]) if f > 0 else None)
    for name, p in sub_probability.items():
        results["E2"][name][str(f)] = scores(cfg, sub_true, p.argmax(1))
        print("E2", f, name, round(results["E2"][name][str(f)]["kappa"], 4), flush=True)
    (OUT / "e2_all.json").write_text(json.dumps(results, indent=1))
print("done", flush=True)
