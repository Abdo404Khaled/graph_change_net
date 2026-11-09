"""Segment graph G of every scored test epoch (fold 0), for the stage-specificity analysis.

Runs the trained GraphChangeNet (classification weights) over the 4,462 scored test epochs,
computes the window's segment graph with the model's own ``sequence.edge_weights`` from the
Level-1 representations, and saves the strictly lower triangle (s < t) of each epoch's G with
its stage and night. No training, no model change. Output ``segment_graphs.npz``:
``G`` [N, 435] (float32), ``stage`` [N], ``subject`` [N]. argv: scaler npz, output npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

SCALER, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUN = REPO / "runs/exp15_final_fold0"


def main():
    """Compute and save the per-epoch segment graphs."""
    cfg = load_run_config(RUN, {})
    split = json.loads((RUN / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(cfg)
    scaler = np.load(SCALER)
    single = data.EpochDataset(cfg, arrays, split["test"], scaler["center"], scaler["scale"], "test", 1)
    device = cfg.device
    model = GraphChangeNet(cfg, len(channels), coords).to(device)
    load_weights(model, RUN / "classify_best.pt", device)
    model.eval()
    lower = np.tril_indices(cfg.data.n_segments, -1)
    samples = cfg.data.n_segments * cfg.data.segment_len
    graphs, stages, subjects, batch, meta = [], [], [], [], []

    def flush():
        """Run the buffered epochs through Level 1 and the segment-graph construction."""
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            hidden, _, _ = model.backbone(x)
            weights = model.sequence.edge_weights(hidden).float().cpu().numpy()
        for w, (stage, subject) in zip(weights, meta):
            graphs.append(w[lower]); stages.append(stage); subjects.append(subject)
        batch.clear(); meta.clear()

    for item in range(len(single)):
        stage = int(single.epochs[single.subject[item]]["stage"][single.row[item]])
        if stage < 0:
            continue
        x, _ = single[item]
        batch.append(x.reshape(-1, len(channels), samples)[-1])
        meta.append((stage, int(single.subject[item])))
        if len(batch) == 32:
            flush()
    flush()
    np.savez(OUT, G=np.array(graphs, dtype=np.float32), stage=np.array(stages), subject=np.array(subjects))
    print(len(graphs), np.bincount(stages))


if __name__ == "__main__":
    main()
