"""Full-test predictions of the fold-0 runs on their own caches (wideband, raw, electrodes_19), for the bootstrap.

Each run's test loader is rebuilt from its own config.yaml and split; scoring as e2_all.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from baselines.common.ours import item_keys  # noqa: E402
from baselines.common.reference import scored_epochs  # noqa: E402
from baselines.common.scoring import save_predictions, score_run  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

OUT = Path(sys.argv[1]); (OUT / "predictions").mkdir(parents=True, exist_ok=True)
results = {}
for name in ["wideband", "raw", "electrodes_19", "wideband_cluster"]:
    run_dir = REPO / f"ablations/results/{name}_fold0"
    cfg = load_run_config(run_dir, {})
    split = json.loads((run_dir / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(cfg)
    test = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)["test"]
    model = GraphChangeNet(cfg, len(channels), coords).to(cfg.device)
    load_weights(model, run_dir / "classify_best.pt", cfg.device)
    model.eval()
    probability = []
    with torch.no_grad():
        for x, y in test:
            keep = (y.flatten() >= 0).numpy()
            probability.append(model(x.to(cfg.device))[0].flatten(0, -2).float().softmax(-1).cpu().numpy()[keep])
    p = np.concatenate(probability)
    names, onsets = item_keys(test, split["test"])
    path = OUT / "predictions" / f"{name}.npz"
    save_predictions(path, np.array(names), np.array(onsets), p.argmax(1), p)
    result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
    reported = json.loads((run_dir / "metrics.json").read_text())["test"]["kappa"]
    results[name] = {"kappa": result["test"]["kappa"], "macro_f1": result["test"]["macro_f1"], "reported_kappa": reported}
    print(name, round(result["test"]["kappa"], 4), "reported", round(reported, 4), flush=True)
    (OUT / "pred_rest.json").write_text(json.dumps(results, indent=1))
print("done", flush=True)
