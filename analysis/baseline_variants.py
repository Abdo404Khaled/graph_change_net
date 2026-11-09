"""The baselines trained under our configuration (fold 0, 0.5-45 Hz cache), scored on other inputs.

No retraining. Each network is built by its own runner exactly as for training (the call to
run_ours / run_graph_ours is intercepted), loaded from baselines/results/<model>/ours/
classify_best.pt (graph models: the DE z-score buffers come with it), and scored on the 5
fold-0 test nights with the input variants already built for Exp 15 (variants/<night>_<i>.npy:
same epochs, blocks, padding and common average; only the filter differs). The scaler stays the
one fitted on the filtered training nights. Scoring: baselines.common.scoring.score_run on all
4,462 scored test epochs, as for the reported results.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.common.ours import OurProtocolModel, item_keys, our_loaders  # noqa: E402
from baselines.common.reference import load_fold, load_reference_config, scored_epochs  # noqa: E402
from baselines.common.scoring import save_predictions, score_run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VARIANT_DIR = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["anysleep", "usleep", "deepsleepnet", "cbramod", "graphsleepnet",
          "graphsleepnet_distance", "mstgcn", "productgraphsleepnet"]
VARIANTS = ["raw (no band-pass, no notch)", "0.5 Hz high-pass + notches, no low-pass",
            "0.5-30 Hz", "0.5-20 Hz"]
settings = yaml.safe_load((REPO / "baselines/settings.yaml").read_text())
settings = {**settings, "reference_run": str(REPO / settings["reference_run"]),
            "fold": str(REPO / settings["fold"])}


class Captured(Exception):
    pass


def capture(network, *args, **kwargs):
    raise Captured(network)


def build(name):
    runner = importlib.import_module(f"baselines.{name}.runner")
    for attribute in ("run_ours", "run_graph_ours"):
        if hasattr(runner, attribute):
            setattr(runner, attribute, capture)
    spec = yaml.safe_load((REPO / f"baselines/{name}/ours.yaml").read_text())
    try:
        runner.run(spec, settings, OUT / "unused" / name, False)
    except Captured as caught:
        network = caught.args[0]
    model = OurProtocolModel(network)
    state = torch.load(REPO / f"baselines/results/{name}/ours/classify_best.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to("cuda").eval()


cfg = load_reference_config(Path(settings["reference_run"]))
split = load_fold(Path(settings["fold"]))
loaders = our_loaders(cfg, split)
test = loaders["test"]
names, onsets = item_keys(test, split["test"])
table = scored_epochs(cfg, split["test"])
models = {}
for name in MODELS:
    models[name] = build(name)
    print("built", name, sum(p.numel() for p in models[name].parameters()), flush=True)

original = [epochs["X"] for epochs in test.dataset.epochs]
results = {name: {} for name in MODELS}
for condition in ["our pipeline (0.5-45 Hz + 60 Hz notch)"] + VARIANTS:
    if condition in VARIANTS:
        index = VARIANTS.index(condition)
        for i, subject in enumerate(split["test"]):
            test.dataset.epochs[i]["X"] = np.load(VARIANT_DIR / f"{subject}_{index}.npy", mmap_mode="r")
    else:
        for i, x in enumerate(original):
            test.dataset.epochs[i]["X"] = x
    probability = {name: [] for name in MODELS}
    with torch.no_grad():
        for x, y in test:
            x = x.to("cuda")
            keep = (y.flatten() >= 0).numpy()
            for name, model in models.items():
                logits = model(x)[0]
                probability[name].append(logits.flatten(0, -2).float().softmax(-1).cpu().numpy()[keep])
    for name in MODELS:
        p = np.concatenate(probability[name])
        folder = OUT / name / str(VARIANTS.index(condition) if condition in VARIANTS else "reference")
        folder.mkdir(parents=True, exist_ok=True)
        save_predictions(folder / "predictions.npz", np.array(names), np.array(onsets), p.argmax(-1), p)
        result = score_run(cfg, table, folder / "predictions.npz")
        results[name][condition] = {k: result["test"][k] for k in ("kappa", "macro_f1", "accuracy")}
        print(condition, name, {k: round(v, 4) for k, v in results[name][condition].items()}, flush=True)
    (OUT / "baseline_variants.json").write_text(json.dumps(results, indent=1))
print("done", flush=True)
