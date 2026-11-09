"""Significance of the stage-1 differences between Exp 15 and every fold-0 ablation.

Every run with a stage 1 on the band-passed input of fold 0 shares the split and the scaler, so the
test epochs are identical. Each run's stage-1 weights (``pretrain_best.pt``, the weights the
reconstruction is scored with) are evaluated on every test epoch under four hiding regimes with the
same masks for every run (argv[3:] restricts the runs to those names, Exp 15 always included; a run
without ``metrics.json``, such as the T1 run whose job ended before writing it, is accepted when its
configured split equals that of Exp 15); the nearest visible electrode is scored on the same
clustered masks as a model-free baseline:

- ``stage1``: the training mask of Exp 15 (random cells plus whole electrodes, ``train.sample_mask``,
  seeded per batch); relative reconstruction loss (smooth L1 against predicting zeros) and relative
  spectral loss (``train.spectral_loss`` against the spectrum with the hidden cells set to zero);
- ``k0`` / ``k5`` / ``k10``: four centre electrodes per epoch hidden with their k nearest
  neighbours for the whole epoch, variance explained of the centres (as ``cluster_all.py``).

Per epoch the sufficient statistics are stored, then a paired two-stage bootstrap (nights with
replacement, then epochs within each night; the same draw for both runs) gives the CI and two-sided p
of each difference (ablation − Exp 15), Holm-corrected within each metric. B and the seed come from
the command line. argv: output JSON path, number of bootstrap draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from baselines.common.ours import item_keys  # noqa: E402
from config import load_run_config  # noqa: E402
from evaluate import nearest_visible  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402
from train import sample_mask, spectral_loss  # noqa: E402

RESULTS = REPO / "ablations/results"
RUNS = {"Exp 15": REPO / "runs/exp15_final_fold0",
        "stable_only": RESULTS / "stable_only_fold0",
        "no_memory": RESULTS / "no_memory_fold0",
        "epoch_graph": RESULTS / "epoch_graph_fold0",
        "distance_graph": RESULTS / "distance_graph_fold0",
        "no_segment_graph": RESULTS / "no_segment_graph_fold0",
        "no_electrode_graph": RESULTS / "no_electrode_graph_fold0",
        "no_graph": RESULTS / "no_graph_fold0",
        "cluster_mask": RESULTS / "cluster_mask_fold0",
        "15b no context": REPO / "runs/exp15b_no_context_fold0",
        "15c reconstruction only": REPO / "runs/exp15c_reconstruction_only_fold0",
        "unsigned_graph": RESULTS / "unsigned_graph_fold0"}
KS = (0, 5, 10)
CENTRES = 4


def epoch_statistics(model: GraphChangeNet, cfg, batches: list, order: np.ndarray,
                     nearest: bool) -> dict:
    """Per-epoch sufficient statistics of every hiding regime for one run.

    Parameters
    ----------
    model : GraphChangeNet
        Stage-1 weights, evaluation mode.
    cfg : Config
        Settings of Exp 15 (masks are drawn with them for every run).
    batches : tuple
        ``(loader, centres)``: the test loader (fixed order) and, per batch, the centre electrodes
        ``[B, CENTRES]``; the mask seed of batch ``i`` is ``1000 + i``.
    order : np.ndarray
        ``[C, C]`` electrodes sorted by distance from each electrode (itself first).
    nearest : bool
        Also score the nearest visible electrode on the clustered masks (``near_sse_k``).

    Returns
    -------
    dict
        Arrays over epochs: ``recon_model``, ``recon_zero`` (smooth-L1 sums over hidden cells),
        ``spectral`` (relative spectral loss), and per k ``sse_k``, ``sum_k``, ``square_k``,
        ``count_k`` of the centre cells.
    """
    device = cfg.device
    out = {name: [] for name in ("recon_model", "recon_zero", "spectral")}
    for k in KS:
        for name in ("sse", "sum", "square", "count") + (("near_sse",) if nearest else ()):
            out[f"{name}_{k}"] = []
    with torch.no_grad():
        loader, all_centres = batches
        for index, (x, *_) in enumerate(loader):
            centres, seed = all_centres[index], 1000 + index
            x = x.to(device)
            target = model.segments(x)
            batch, segments, channels, _ = target.shape
            torch.manual_seed(seed)
            mask = sample_mask(cfg, batch, channels, device)
            rebuilt = model.pretrain_outputs(x, mask)[0]
            for item in range(batch):
                hidden = mask[item]
                guess, truth = rebuilt[item][hidden], target[item][hidden]
                out["recon_model"].append(F.smooth_l1_loss(guess, truth, reduction="sum").item())
                out["recon_zero"].append(F.smooth_l1_loss(torch.zeros_like(truth), truth,
                                                          reduction="sum").item())
                one = slice(item, item + 1)
                value = spectral_loss(cfg, rebuilt[one], target[one], mask[one])
                trivial = spectral_loss(cfg, torch.zeros_like(rebuilt[one]), target[one], mask[one])
                out["spectral"].append((value / trivial.clamp(min=1e-8)).item())
            for k in KS:
                hide = torch.zeros(batch, channels, dtype=torch.bool, device=device)
                score = torch.zeros(batch, channels, dtype=torch.bool, device=device)
                for item in range(batch):
                    for centre in centres[item]:
                        hide[item, torch.as_tensor(order[centre, :k + 1], device=device)] = True
                        score[item, int(centre)] = True
                cluster = hide.unsqueeze(1).expand(batch, segments, channels)
                prediction = model.reconstruct(x, cluster)[0]
                near = nearest_visible(target, cluster, order[:, 1:]) if nearest else None
                for item in range(batch):
                    cells = score[item].unsqueeze(0).expand(segments, channels)
                    guess, truth = prediction[item][cells], target[item][cells]
                    out[f"sse_{k}"].append(((guess - truth) ** 2).sum().item())
                    out[f"sum_{k}"].append(truth.sum().item())
                    out[f"square_{k}"].append((truth ** 2).sum().item())
                    out[f"count_{k}"].append(truth.numel())
                    if nearest:
                        out[f"near_sse_{k}"].append(((near[item][cells] - truth) ** 2).sum().item())
    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def metrics(stats: dict, index: np.ndarray) -> dict:
    """Pooled stage-1 metrics of the epochs ``index``.

    Parameters
    ----------
    stats : dict
        Output of :func:`epoch_statistics`.
    index : np.ndarray
        Epoch indices (with repeats for a bootstrap draw).

    Returns
    -------
    dict
        ``reconstruction`` (relative loss, lower is better), ``spectral`` (relative, lower is
        better) and ``ve_k0`` / ``ve_k5`` / ``ve_k10`` (higher is better).
    """
    result = {"reconstruction": stats["recon_model"][index].sum() / stats["recon_zero"][index].sum(),
              "spectral": stats["spectral"][index].mean()}
    for k in KS:
        n = stats[f"count_{k}"][index].sum()
        total = stats[f"sum_{k}"][index].sum()
        variance = stats[f"square_{k}"][index].sum() - total ** 2 / n
        result[f"ve_k{k}"] = 1 - stats[f"sse_{k}"][index].sum() / variance
    return result


def holm(p_values: dict) -> dict:
    """Holm-adjusted p-values.

    Parameters
    ----------
    p_values : dict
        Name to raw p-value.

    Returns
    -------
    dict
        Name to adjusted p-value.
    """
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def main():
    """Evaluate every run, bootstrap every difference against Exp 15 and write the table."""
    out_path, draws = Path(sys.argv[1]), int(sys.argv[2])
    base = load_run_config(RUNS["Exp 15"], {})
    split = json.loads((RUNS["Exp 15"] / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(base)
    loaders = data.build_epoch_loaders(base, {s: arrays[s] for s in sum(split.values(), [])}, split)
    test = loaders["test_single"]
    nights = np.array(item_keys(test, split["test"])[0])
    order = NearestNeighbors(n_neighbors=len(coords)).fit(coords).kneighbors(
        coords, return_distance=False)
    generator = np.random.default_rng(0)
    sizes = [min(test.batch_size, len(test.dataset) - start)
             for start in range(0, len(test.dataset), test.batch_size)]
    batches = (test, [np.stack([generator.choice(len(channels), CENTRES, replace=False)
                                for _ in range(size)]) for size in sizes])
    stats = {}
    selected = {name: path for name, path in RUNS.items()
                if not sys.argv[3:] or name == "Exp 15" or name in sys.argv[3:]}
    for name, run_dir in selected.items():
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            same = json.loads(metrics_path.read_text())["split"] == split
        else:
            same = load_run_config(run_dir, {}).split == base.split
        if not same:
            raise ValueError(f"{name} has another split than Exp 15")
        cfg = load_run_config(run_dir, {})
        model = GraphChangeNet(cfg, len(channels), coords).to(base.device)
        load_weights(model, run_dir / "pretrain_best.pt", base.device)
        stats[name] = epoch_statistics(model.eval(), base, batches, order, name == "Exp 15")
        print(name, {k: round(float(v), 4) for k, v in
                     metrics(stats[name], np.arange(len(nights))).items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    groups = [np.flatnonzero(nights == night) for night in np.unique(nights)]
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(draws):
        chosen = rng.integers(len(groups), size=len(groups))
        samples.append(np.concatenate([rng.choice(groups[g], size=len(groups[g])) for g in chosen]))
    every = np.arange(len(nights))
    reference = metrics(stats["Exp 15"], every)
    reference_draws = [metrics(stats["Exp 15"], s) for s in samples]
    table = {"n_epochs": int(len(nights)), "nights": sorted(set(nights.tolist())), "draws": draws,
             "Exp 15": reference, "variants": {}}
    table["nearest_visible"] = {}
    reference_stats = stats["Exp 15"]
    for k in KS:
        def near_ve(index):
            """VE of the nearest visible electrode on the epochs ``index`` at this hiding level."""
            n = reference_stats[f"count_{k}"][index].sum()
            total = reference_stats[f"sum_{k}"][index].sum()
            variance = reference_stats[f"square_{k}"][index].sum() - total ** 2 / n
            return 1 - reference_stats[f"near_sse_{k}"][index].sum() / variance
        point = near_ve(every)
        differences = np.array([r[f"ve_k{k}"] - near_ve(draw) for draw, r in zip(samples, reference_draws)])
        table["nearest_visible"][f"ve_k{k}"] = {
            "value": float(point), "exp15_minus_nearest": float(reference[f"ve_k{k}"] - point),
            "ci": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
            "p": float(min(1.0, 2 * min((differences <= 0).mean(), (differences >= 0).mean())))}
    print("nearest visible", {k: round(v["value"], 4) for k, v in table["nearest_visible"].items()},
          flush=True)
    for name in selected:
        if name == "Exp 15":
            continue
        point = metrics(stats[name], every)
        rows = {}
        for metric, value in point.items():
            differences = np.array([metrics(stats[name], s)[metric] - r[metric]
                                    for s, r in zip(samples, reference_draws)])
            p = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
            rows[metric] = {"value": value, "delta": value - reference[metric],
                            "ci": [float(np.percentile(differences, 2.5)),
                                   float(np.percentile(differences, 97.5))],
                            "p": float(min(1.0, p))}
        table["variants"][name] = rows
    for metric in reference:
        adjusted = holm({name: rows[metric]["p"] for name, rows in table["variants"].items()})
        for name, value in adjusted.items():
            table["variants"][name][metric]["p_holm"] = value
    out_path.write_text(json.dumps(table, indent=1, default=float))
    for name, rows in table["variants"].items():
        print(f"{name:<24}" + "  ".join(f"{m} {r['delta']:+.4f} (p_holm {r['p_holm']:.3f})"
                                          for m, r in rows.items()))


if __name__ == "__main__":
    main()
