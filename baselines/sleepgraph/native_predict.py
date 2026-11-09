"""Predict every test epoch of a native sleep-graph-benchmark run, and keep its graphs.

Run as a separate process with the upstream repository first on ``sys.path``
(``baselines.sleepgraph.native.predict`` does this): upstream's packages are
called ``data``, ``models``, ``training`` and ``evaluation``, and ``data``
would collide with our own module. Nothing here imports our code.

Upstream trains with ``pad_mode=drop``, which gives no prediction to the
first and last ``window // 2`` epochs of every continuous segment. Our scorer
needs every scored epoch, so the selected checkpoint is run again over the
test recordings with upstream's ``pad_mode=edge`` windows (the edge epoch
repeated). For the windows that ``drop`` also has, the predictions must equal
upstream's own ``oof_predictions.npz``; this is checked and reported.

MSTGCN and the CNN-2S runs (``*_cnn2s``, upstream's ``--condition cnn_2s``:
the shared four-block waveform CNN trained first, then the graph backbone on
its frozen per-epoch features) read the node features the run itself saved
in ``fold_0/features.npz``, so the windows are rebuilt from exactly the
features the selected backbone was trained and tested on.

With ``--causal`` (runs trained by ``causal_launcher.py``) every window ends at
its scored epoch instead of being centred on it, so the agreement check with
upstream's own predictions applies to the same past-only windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.recordings import context_window_indices, iter_window_centers, load_recordings
from data.splits import load_splits
from models.graphsleepnet import GraphSleepNetDE, ReleasedGraphSleepNetBackbone
from models.mstgcn import ReleasedMSTGCN, released_static_polynomials
from models.productgraphsleepnet import (ProductGraphSleepNetDE,
                                         ReleasedProductGraphSleepNetBackbone)

WINDOWS = {"graphsleepnet": 5, "graphsleepnet_distance": 5, "mstgcn": 5,
           "productgraphsleepnet": 9, "graphsleepnet_cnn2s": 5,
           "productgraphsleepnet_cnn2s": 9}
FROZEN_FEATURES = ("mstgcn", "graphsleepnet_cnn2s", "productgraphsleepnet_cnn2s")


def windows(record, window: int, pad_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Window centres and their context indices, as upstream's datasets build them.

    Parameters
    ----------
    record : Recording
        One upstream recording.
    window : int
        Epochs per window.
    pad_mode : str
        ``drop`` or ``edge``.

    Returns
    -------
    centres : np.ndarray
        ``[N]`` centre epochs.
    indices : np.ndarray
        ``[N, window]`` context epochs.
    """
    centres = np.array(list(iter_window_centers(record, window_size=window, pad_mode=pad_mode)),
                       dtype=np.int64)
    indices = np.stack([context_window_indices(record, int(c), window_size=window,
                                               pad_mode=pad_mode) for c in centres])
    return centres, indices


def build(kind: str, checkpoint: dict, vertices: int, distance: np.ndarray | None
          ) -> torch.nn.Module:
    """The upstream network of a native run, with its selected weights.

    Parameters
    ----------
    kind : str
        Model folder name.
    checkpoint : dict
        Upstream ``best_model.pt``.
    vertices : int
        Channels.
    distance : np.ndarray or None
        Fixed adjacency of ``graphsleepnet_distance``.

    Returns
    -------
    torch.nn.Module
        Network in evaluation mode.
    """
    state = checkpoint["model_state_dict"]
    if kind == "graphsleepnet_cnn2s":
        network = ReleasedGraphSleepNetBackbone(vertices, timesteps=5)
    elif kind == "productgraphsleepnet_cnn2s":
        args = checkpoint["args"]
        network = ReleasedProductGraphSleepNetBackbone(
            vertices, timesteps=9, gru_cell=args["gru_cell"], heads=args["attention_heads"],
            dropout=args["dropout"], graph_alpha=0.0)
    elif kind in ("graphsleepnet", "graphsleepnet_distance"):
        network = GraphSleepNetDE(vertices, state["de_mean"], state["de_std"], timesteps=5)
    elif kind == "productgraphsleepnet":
        args = checkpoint["args"]
        network = ProductGraphSleepNetDE(
            vertices, state["de_mean"], state["de_std"], timesteps=9, gru_cell=args["gru_cell"],
            heads=args["attention_heads"], dropout=args["dropout"], graph_alpha=0.0)
    else:
        polynomials = released_static_polynomials(np.asarray(checkpoint["static_adjacency"]), 3)
        domains = state["domain_head.2.weight"].shape[0]
        network = ReleasedMSTGCN(vertices, polynomials, domains)
    network.load_state_dict(state)
    if distance is not None:
        network.distance = torch.as_tensor(distance, dtype=torch.float32)
    return network.eval()


@torch.no_grad()
def forward(kind: str, network: torch.nn.Module, inputs: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and the electrode graph of the centre epoch of each window.

    Parameters
    ----------
    kind : str
        Model folder name.
    network : torch.nn.Module
        Upstream network.
    inputs : torch.Tensor
        ``[B, window, V, F]`` DE features or MSTGCN features.

    Returns
    -------
    logits : torch.Tensor
        ``[B, 5]``.
    graph : torch.Tensor
        ``[B, V, V]``: GraphSleepNet's learned (or fixed) adjacency, the
        product model's spatial graph, MSTGCN's learned graph of the centre
        epoch.
    """
    if kind in ("graphsleepnet", "graphsleepnet_cnn2s"):
        out = network(inputs)
        return out["logits"], out["adjacency"]
    if kind == "graphsleepnet_distance":
        normalised = (inputs - network.de_mean) / network.de_std
        graph = network.distance.to(inputs).expand(inputs.shape[0], -1, -1)
        out = network.backbone(normalised, adjacency_override=graph)
        return out["logits"], out["adjacency"]
    if kind in ("productgraphsleepnet", "productgraphsleepnet_cnn2s"):
        out = network(inputs)
        return out["logits"], out["spatial_graph"]
    out = network(inputs)
    learned = network.block.graph_learn(inputs)[0]
    return out["logits"], learned[:, inputs.shape[1] // 2]


def main() -> None:
    """Command line; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distance", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--causal", action="store_true",
                        help="past-only windows, for runs trained by causal_launcher.py")
    args = parser.parse_args()
    if args.causal:
        sys.path.insert(1, str(Path(__file__).resolve().parent))
        import causal_windows
        causal_windows.install()
        globals()["iter_window_centers"] = causal_windows.causal_window_centers
        globals()["context_window_indices"] = causal_windows.causal_context_indices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = args.run_dir / "fold_0"
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location="cpu", weights_only=False)
    split = load_splits(args.split_file, args.data_dir)["folds"]["0"]
    files = [str(Path(path).resolve()) for path in split["test_files"]]
    frozen = args.kind in FROZEN_FEATURES
    records = load_recordings(files, input_scaling=checkpoint["args"].get(
        "native_input_scaling", checkpoint["args"].get("input_scaling", "none")),
        representation="raw" if frozen else "de")
    if frozen:
        stored = np.load(fold_dir / "features.npz", allow_pickle=True)
    vertices = len(checkpoint["channels"])
    distance = np.load(args.distance) if args.distance else None
    network = build(args.kind, checkpoint, vertices, distance).to(device)

    rows = {key: [] for key in ("recording_id", "epoch_index", "label", "probability", "graph")}
    agreement = []
    upstream = np.load(fold_dir / "oof_predictions.npz", allow_pickle=True)
    upstream_prediction = {(str(r), int(e)): int(p) for r, e, p in zip(
        upstream["recording_id"], upstream["epoch_index"], upstream["prediction"])}
    for path in files:
        record = records[path]
        features = torch.as_tensor(stored[Path(path).stem]) if frozen else record.de
        centres, indices = windows(record, WINDOWS[args.kind], "edge")
        name = Path(path).stem
        for start in range(0, len(centres), args.batch_size):
            batch = torch.as_tensor(indices[start:start + args.batch_size])
            logits, graph = forward(args.kind, network, features[batch].float().to(device))
            probability = logits.softmax(-1).cpu().numpy()
            for row, centre in enumerate(centres[start:start + args.batch_size]):
                epoch = int(record.original_epoch_index[centre])
                rows["recording_id"].append(name)
                rows["epoch_index"].append(epoch)
                rows["label"].append(int(record.y[centre]))
                rows["probability"].append(probability[row])
                rows["graph"].append(graph[row].float().cpu().numpy().astype(np.float16))
                if (name, epoch) in upstream_prediction:
                    agreement.append(upstream_prediction[(name, epoch)]
                                     == int(probability[row].argmax()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"epochs": len(rows["epoch_index"]), "upstream_epochs": len(agreement),
               "agreement_with_upstream": float(np.mean(agreement)) if agreement else None,
               "parameters": sum(p.numel() for p in network.parameters())}
    np.savez_compressed(args.out, **{key: np.asarray(value) for key, value in rows.items()},
                        channels=np.asarray(checkpoint["channels"]),
                        summary=np.asarray(json.dumps(summary)))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
