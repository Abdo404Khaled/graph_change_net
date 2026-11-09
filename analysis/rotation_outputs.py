"""Full-test predictions of the Exp 15 rotation runs (7 groups of 4 nights, every night tested once).

Each run ``runs/exp15_rotation_rot<k>`` is evaluated through its own loaders, so the scaler fitted on
that run's 20 training nights is used, and its kappa is checked against its own metrics.json.
Predictions are saved in the scorer's format (subject, onset) for the run's 4 test nights, so the
7 files together cover all 28 nights and can be paired epoch by epoch with any baseline scored on the
same rotation folds. argv: output folder. Runs that are not present yet, and runs already predicted
(listed in ``rotation.json`` with their file), are skipped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from baselines.common.reference import scored_epochs  # noqa: E402
from baselines.common.ours import item_keys  # noqa: E402
from baselines.common.scoring import save_predictions, score_run  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)


def main():
    """Predict the test nights of every finished rotation run and save the predictions."""
    runs = [REPO / f"runs/exp15_rotation_rot{k}" for k in range(7)]
    runs = [r for r in runs if (r / "classify_best.pt").exists() and (r / "metrics.json").exists()]
    if not runs:
        raise SystemExit("no finished rotation run found under runs/")
    first = load_run_config(runs[0], {})
    arrays, channels, coords = data.load_dataset(first)
    summary_path = OUT / "rotation.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for run_dir in runs:
        if run_dir.name.rsplit("_", 1)[1] in summary and (OUT / f"{run_dir.name.rsplit('_', 1)[1]}.npz").exists():
            continue
        cfg = load_run_config(run_dir, {})
        metrics = json.loads((run_dir / "metrics.json").read_text())
        split = metrics["split"]
        loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)
        test = loaders["test"]
        names, onsets = item_keys(test, split["test"])
        model = GraphChangeNet(cfg, len(channels), coords).to(cfg.device)
        load_weights(model, run_dir / "classify_best.pt", cfg.device)
        model.eval()
        probability = []
        with torch.no_grad():
            for x, y in test:
                keep = (y.flatten() >= 0).numpy()
                logits = model(x.to(cfg.device))[0]
                probability.append(logits.flatten(0, -2).float().softmax(-1).cpu().numpy()[keep])
        probability = np.concatenate(probability)
        name = run_dir.name.rsplit("_", 1)[1]
        path = OUT / f"{name}.npz"
        save_predictions(path, np.array(names), np.array(onsets), probability.argmax(1), probability)
        result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
        summary[name] = {"test": split["test"], "kappa": result["test"]["kappa"],
                         "macro_f1": result["test"]["macro_f1"],
                         "reported_kappa": metrics["test"]["kappa"],
                         "reported_macro_f1": metrics["test"]["macro_f1"],
                         "parameters": metrics["parameters"]}
        print(name, summary[name], flush=True)
        del model
        torch.cuda.empty_cache()
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
