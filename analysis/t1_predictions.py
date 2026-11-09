"""Full-test predictions of the run trained without edge signs (T1, fold 0), no training.

The training job of this run ended before writing ``metrics.json``, so its split is taken from Exp 15
fold 0, which its configuration shares (same fold, seed and split settings; checked). The run is
evaluated through its own loaders and saved in the scorer's format next to the other fold-0
prediction sets, as ``unsigned_graph.npz``, for the paired bootstrap of ``stats.py``.
argv: output folder (the e2_all predictions folder).
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


def main():
    """Predict the test nights with the T1 weights, save and score them."""
    out = Path(sys.argv[1])
    run_dir = REPO / "ablations/results/unsigned_graph_fold0"
    reference = REPO / "runs/exp15_final_fold0"
    cfg = load_run_config(run_dir, {})
    if cfg.split != load_run_config(reference, {}).split:
        raise ValueError("T1 and Exp 15 fold 0 have different split settings")
    split = json.loads((reference / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(cfg)
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
    path = out / "unsigned_graph.npz"
    save_predictions(path, np.array(names), np.array(onsets), probability.argmax(1), probability)
    result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
    print(json.dumps({"kappa": result["test"]["kappa"], "macro_f1": result["test"]["macro_f1"],
                      "f1_N1": result["test"]["f1_N1"]}))


if __name__ == "__main__":
    main()
