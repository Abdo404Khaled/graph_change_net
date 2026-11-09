"""Stage-specific graphs at both levels (fold 0, all 4,462 scored test epochs): figure and statistics.

Electrode level: the epoch-mean graph ``A`` of every test epoch (``runs/exp15_final_fold0/graphs.npz``);
for each stage the deviation of its mean from the mean over all epochs, with a per-edge
permutation test that shuffles stage labels within each night (200 permutations; smallest
attainable p = 1/201). Drawn in the three orthogonal projections, one row per stage: each
stage's strongest significant increases (red) and decreases (blue). Segment level: the segment
graph G of every test epoch (``results/segment_graphs.npz`` from ``segment_stage.py``), its stage
deviations, the mean edge by lag per stage, and the same global test (its figure is drawn by
``segment_figure.py``). Change: ``delta_norm``
(Frobenius norm of Delta_t per second) per stage. Global statistic per level: the between-stage
dispersion of the stage means (weighted by stage share), compared with its within-night
label-shuffle null. argv: output folder of the figures; ``stage_specificity.json`` goes to ``analysis/results``.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PREVIEW = REPO / "analysis/results/figures"
PREVIEW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from baselines.common.reference import load_reference_config  # noqa: E402
from baselines.graph_tests import load_run, montage, stages_of  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUN = REPO / "runs/exp15_final_fold0"
NOTES = REPO / "analysis/results"
STAGES = ["W", "N1", "N2", "N3", "R"]
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
POS, NEG, INK, INK2, MUTED, BASE, PAPER = "#e34948", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#f0efec"
SIGNED = LinearSegmentedColormap.from_list("signed", [NEG, PAPER, POS])
PERMS = 200
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
rng = np.random.default_rng(0)


def shuffle_within(stage, subject):
    """Permute stage labels within each night.

    Parameters
    ----------
    stage : numpy.ndarray
        Stage per epoch.
    subject : numpy.ndarray
        Night per epoch.

    Returns
    -------
    numpy.ndarray
        Shuffled labels.
    """
    shuffled = stage.copy()
    for night in np.unique(subject):
        index = np.flatnonzero(subject == night)
        shuffled[index] = rng.permutation(stage[index])
    return shuffled


def deviations(values, stage):
    """Stage mean minus overall mean, per stage.

    Parameters
    ----------
    values : numpy.ndarray
        ``[N, E]`` per-epoch features.
    stage : numpy.ndarray
        Stage per epoch.

    Returns
    -------
    numpy.ndarray
        ``[5, E]``.
    """
    mean = values.mean(0)
    return np.array([values[stage == k].mean(0) - mean for k in range(5)])


def test(values, stage, subject):
    """Global and per-feature within-night permutation tests of stage specificity.

    Parameters
    ----------
    values : numpy.ndarray
        ``[N, E]`` per-epoch features.
    stage, subject : numpy.ndarray
        Labels per epoch.

    Returns
    -------
    dict
        Observed deviations, per-feature p-values, global ratio to the null mean and global p.
    """
    share = np.array([(stage == k).mean() for k in range(5)])
    observed = deviations(values, stage)
    statistic = float((share[:, None] * observed ** 2).sum())
    exceed = np.zeros_like(observed)
    null = []
    for _ in range(PERMS):
        permuted = deviations(values, shuffle_within(stage, subject))
        exceed += np.abs(permuted) >= np.abs(observed)
        null.append(float((share[:, None] * permuted ** 2).sum()))
    return {"deviation": observed, "p": (1 + exceed) / (PERMS + 1), "ratio": statistic / float(np.mean(null)),
            "global_p": (1 + sum(n >= statistic for n in null)) / (PERMS + 1)}


def draw(axis, view, coords, rows, cols, weights, top):
    """Draw significant deviation edges of one stage in one projection.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Panel.
    view : callable
        Maps 3-D coordinates to 2-D.
    coords : numpy.ndarray
        ``[C, 3]`` electrode positions.
    rows, cols, weights : numpy.ndarray
        Edges and their signed deviations.
    top : float
        Deviation drawn at full width.
    """
    x, y = view(coords)
    for e in np.argsort(np.abs(weights)):
        r = min(abs(weights[e]) / top, 1.0)
        axis.plot([x[rows[e]], x[cols[e]]], [y[rows[e]], y[cols[e]]], color=POS if weights[e] > 0 else NEG,
                  lw=0.3 + 1.1 * r, alpha=0.35 + 0.6 * r)
    axis.scatter(x, y, s=0.8, color=INK, lw=0, zorder=3)
    axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main():
    """Compute the statistics, draw the figure and write the JSON."""
    cfg = load_reference_config(RUN)
    names, coords = montage(cfg)
    coords = coords - coords.mean(0)
    run = load_run(RUN / "graphs.npz", names)
    stage = stages_of(cfg, run)
    subject = run["subject"]
    delta = np.load(RUN / "graphs.npz")["delta_norm"].astype(np.float32)
    upper = np.triu_indices(len(names), 1)
    edges = run["graph"][:, upper[0], upper[1]]
    electrode = test(edges, stage, subject)
    seg = np.load(NOTES / "segment_graphs.npz")
    segment = test(seg["G"], seg["stage"], seg["subject"].astype(str))
    lower = np.tril_indices(30, -1)
    lag = lower[0] - lower[1]
    settled = delta[:, 14:].mean(1)
    level = test(settled[:, None], stage, subject)
    strength = np.abs(run["graph"]).sum(-1)
    nodes = test(strength, stage, subject)
    folds = {}
    for k in range(5):
        path = REPO / f"runs/exp15_final_fold{k}/graphs.npz"
        if path.exists():
            fold_run = load_run(path, names)
            folds[k] = {"stable": np.load(path)["stable"][upper], "deviation": deviations(fold_run["graph"][:, upper[0], upper[1]], stages_of(cfg, fold_run))}
    pairs = [(a, b) for a in folds for b in folds if a < b]
    stability = {"folds": sorted(folds),
                 "stable_graph_r": [float(np.corrcoef(folds[a]["stable"], folds[b]["stable"])[0, 1]) for a, b in pairs],
                 "stage_deviation_r": {name: [float(np.corrcoef(folds[a]["deviation"][k], folds[b]["deviation"][k])[0, 1]) for a, b in pairs]
                                       for k, name in enumerate(STAGES)}}

    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    mean_distance = distance[upper].mean()
    low = coords[:, 2] < np.quantile(coords[:, 2], 0.25)
    summary = {"electrode": {"ratio": electrode["ratio"], "global_p": electrode["global_p"], "stages": {}},
               "segment": {"ratio": segment["ratio"], "global_p": segment["global_p"], "stages": {}},
               "change_level": {"ratio": level["ratio"], "global_p": level["global_p"], "stages": {}},
               "node_strength": {"ratio": nodes["ratio"], "global_p": nodes["global_p"],
                                 "significant_nodes": {name: int((nodes["p"][k] <= 1 / (PERMS + 1)).sum()) for k, name in enumerate(STAGES)}},
               "fold_stability": stability}
    for k, name in enumerate(STAGES):
        dev, p = electrode["deviation"][k], electrode["p"][k]
        significant = np.flatnonzero(p <= 1 / (PERMS + 1))
        entry = {"epochs": int((stage == k).sum()), "significant_edges": int(len(significant))}
        for sign, label in ((1, "stronger"), (-1, "weaker")):
            chosen = significant[sign * dev[significant] > 0]
            chosen = chosen[np.argsort(-np.abs(dev[chosen]))][:20]
            entry[label] = {"count": int((sign * dev[significant] > 0).sum()),
                            "mean_length": float(distance[upper[0][chosen], upper[1][chosen]].mean() / mean_distance) if len(chosen) else None,
                            "share_with_low_electrode": float((low[upper[0][chosen]] | low[upper[1][chosen]]).mean()) if len(chosen) else None}
        summary["electrode"]["stages"][name] = entry
        sdev = segment["deviation"][k]
        summary["segment"]["stages"][name] = {
            "epochs": int((seg["stage"] == k).sum()),
            "significant_edges": int((segment["p"][k] <= 1 / (PERMS + 1)).sum()),
            "lag1_mean": float(seg["G"][seg["stage"] == k][:, lag == 1].mean()),
            "lag1_deviation": float(sdev[lag == 1].mean()),
            "max_abs_deviation": float(np.abs(sdev).max())}
        values = settled[stage == k]
        summary["change_level"]["stages"][name] = {"mean": float(values.mean()), "se": float(values.std() / np.sqrt(len(values))),
                                                   "per_night": {str(n): float(settled[(stage == k) & (subject == n)].mean())
                                                                 for n in np.unique(subject) if ((stage == k) & (subject == n)).any()}}

    plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7,
                         "legend.fontsize": 6.3})
    views = [("top", lambda q: (-q[:, 1], q[:, 0])), ("side", lambda q: (-q[:, 0], q[:, 2])),
             ("front", lambda q: (q[:, 1], q[:, 2]))]
    xy_top = np.stack([-coords[:, 1], coords[:, 0]], 1)
    top_dev = np.quantile(np.abs(electrode["deviation"]), 0.999)
    node_limit = np.abs(nodes["deviation"]).max()
    from matplotlib.path import Path as MPath
    from matplotlib.patches import Ellipse, PathPatch
    radius = np.linalg.norm(coords, axis=1)
    elevation = np.degrees(np.arcsin(coords[:, 2] / radius))
    inferior = elevation < np.percentile(elevation, 22)
    front = coords[:, 0] / radius
    side_l = coords[:, 1] >= 0
    band = np.where(inferior, 3, np.where(front > 0.35, 0, np.where(front < -0.35, 2, 1)))
    region = band * 2 + (~side_l).astype(int)
    region_names = ["front L", "front R", "centre L", "centre R", "back L", "back R", "bottom L", "bottom R"]

    def head(axis, view, xs, ys):
        cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        w, h = (xs.max() - xs.min()) * 1.16, (ys.max() - ys.min()) * 1.16
        axis.add_patch(Ellipse((cx, cy), w, h, fill=True, fc="#f7f6f2", ec=MUTED, lw=0.7, zorder=0))
        if view == "top":
            axis.add_patch(plt.Polygon([[cx - 0.08 * w, cy + 0.49 * h], [cx, cy + 0.6 * h], [cx + 0.08 * w, cy + 0.49 * h]],
                                       fc="#f7f6f2", ec=MUTED, lw=0.7, zorder=0))
            for sgn in (-1, 1):
                axis.add_patch(Ellipse((cx + sgn * 0.52 * w, cy), 0.08 * w, 0.2 * h, fc="#f7f6f2", ec=MUTED, lw=0.7, zorder=0))
        elif view == "side":
            axis.add_patch(plt.Polygon([[cx - 0.49 * w, cy + 0.05 * h], [cx - 0.6 * w, cy - 0.06 * h], [cx - 0.47 * w, cy - 0.12 * h]],
                                       fc="#f7f6f2", ec=MUTED, lw=0.7, zorder=0))
        else:
            for sgn in (-1, 1):
                axis.add_patch(Ellipse((cx + sgn * 0.52 * w, cy - 0.05 * h), 0.08 * w, 0.22 * h, fc="#f7f6f2", ec=MUTED, lw=0.7, zorder=0))
        axis.set_xlim(cx - 0.68 * w, cx + 0.68 * w); axis.set_ylim(cy - 0.62 * h, cy + 0.66 * h)

    def curved(axis, a, b, colour, width, alpha):
        mid = (a + b) / 2
        normal = np.array([-(b - a)[1], (b - a)[0]])
        control = mid + 0.18 * normal
        path = MPath([a, control, b], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        axis.add_patch(PathPatch(path, fc="none", ec=colour, lw=width, alpha=alpha, capstyle="round", zorder=2))

    views = [("top", lambda q: (-q[:, 1], q[:, 0])), ("side", lambda q: (-q[:, 0], q[:, 2])), ("front", lambda q: (q[:, 1], q[:, 2]))]
    node_limit = np.abs(nodes["deviation"]).max()
    block_limit = 0
    blocks = []
    for k in range(5):
        dev = electrode["deviation"][k]
        full = np.zeros((len(coords), len(coords))); full[upper] = dev; full = full + full.T
        block = np.array([[full[np.ix_(region == a, region == b)][~np.eye(len(coords), dtype=bool)[np.ix_(region == a, region == b)]].mean()
                           for b in range(8)] for a in range(8)])
        blocks.append(block); block_limit = max(block_limit, np.abs(block).max())
    index = {name: i for i, name in enumerate(region_names)}
    pair_defs = [("bottom circ., left$-$right", [("bottom L", "bottom R")]),
                 ("bottom circ.$-$frontal", [(a, b) for a in ("bottom L", "bottom R") for b in ("front L", "front R")]),
                 ("bottom circ.$-$central", [(a, b) for a in ("bottom L", "bottom R") for b in ("centre L", "centre R")]),
                 ("frontal, left$-$right", [("front L", "front R")]),
                 ("front$-$back, same side", [("front L", "back L"), ("front R", "back R")])]
    pairs = {label: [float(np.mean([blocks[k][index[a], index[b]] for a, b in members])) for k in range(5)]
             for label, members in pair_defs}
    summary["region_pairs"] = {label.replace("$-$", "-"): dict(zip(STAGES, values)) for label, values in pairs.items()}

    fig = plt.figure(figsize=(7.16, 2.35))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], left=0.0, right=0.985, top=0.9, bottom=0.17, wspace=0.06)
    heads = outer[0].subgridspec(2, 6, width_ratios=[0.34] + [1] * 5, wspace=0.03, hspace=0.04)
    top_view = views[0][1]
    xs, ys = top_view(coords)
    for row, word in enumerate(["stronger\nthan\naverage", "weaker\nthan\naverage"]):
        label = fig.add_subplot(heads[row, 0]); label.axis("off")
        label.text(0.5, 0.5, word, ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.15)
    for k, name in enumerate(STAGES):
        dev, p = electrode["deviation"][k], electrode["p"][k]
        significant = np.flatnonzero(p <= 1 / (PERMS + 1))
        up, down = significant[dev[significant] > 0], significant[dev[significant] < 0]
        for row, (chosen, colour) in enumerate([(up[np.argsort(-dev[up])][:12], POS), (down[np.argsort(dev[down])][:12], NEG)]):
            axis = fig.add_subplot(heads[row, 1 + k])
            head(axis, "top", xs, ys)
            for e in chosen[np.argsort(np.abs(dev[chosen]))]:
                r = min(abs(dev[e]) / top_dev, 1.0)
                a = np.array([xs[upper[0][e]], ys[upper[0][e]]]); b = np.array([xs[upper[1][e]], ys[upper[1][e]]])
                curved(axis, a, b, colour, 0.6 + 1.6 * r, 0.5 + 0.45 * r)
            axis.scatter(xs, ys, s=2, color=MUTED, lw=0, zorder=3)
            axis.set_aspect("equal"); axis.axis("off")
            if row == 0:
                axis.set_title(f"{name}  ({(stage == k).sum()})", fontsize=7.5, color=STAGE_COLORS[k], weight="bold", pad=1)
    fig.text(0.0, 0.965, "(a) the 12 significant edges per sign that differ most from the average (top view, nose up)",
             fontsize=7, ha="left", va="center")
    dots = fig.add_subplot(outer[1])
    labels = list(pairs)
    for row, label in enumerate(labels):
        dots.axhline(row, color="#e1e0d9", lw=0.5, zorder=0)
        for k in range(5):
            dots.scatter(pairs[label][k], row, s=16, color=STAGE_COLORS[k], edgecolor="white", lw=0.4, zorder=3,
                         label=STAGES[k] if row == 0 else None)
    dots.axvline(0, color=BASE, lw=0.7, zorder=1)
    limit = max(abs(v) for values in pairs.values() for v in values) * 1.12
    dots.set_xlim(-limit, limit)
    for row, label in enumerate(labels):
        dots.text(-limit, row - 0.3, label, fontsize=6.3, color=INK2, ha="left", va="center")
    dots.set_yticks([])
    dots.set_ylim(len(labels) - 0.55, -0.75)
    dots.set_xlabel("mean edge, stage $-$ all", fontsize=6.6, labelpad=1)
    dots.tick_params(axis="x", labelsize=6, length=2, pad=1); dots.tick_params(axis="y", length=0, pad=2)
    dots.spines[["top", "right", "left"]].set_visible(False)
    dots.set_title("(b) by region pair", fontsize=7, loc="left", pad=3)
    dots.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6, handletextpad=0.05,
                columnspacing=0.5, borderaxespad=0, markerscale=0.9)
    fig.savefig(OUT / "fig_stage_graph.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "fig_stage_graph.png", dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    summary["regions"] = {"names": region_names, "blocks": {STAGES[k]: np.round(blocks[k], 4).tolist() for k in range(5)}}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.2)
    t = np.arange(1, 31)
    for k, name in enumerate(STAGES):
        rows = delta[stage == k]
        m, se = rows.mean(0), rows.std(0) / np.sqrt(len(rows))
        ax.plot(t, m, color=STAGE_COLORS[k], lw=1.2, label=name)
        ax.fill_between(t, m - se, m + se, color=STAGE_COLORS[k], alpha=0.2, lw=0)
    ax.axvspan(15, 30, color="#e1e0d9", alpha=0.45, lw=0, zorder=0)
    ax.text(22.5, ax.get_ylim()[1], "settled", ha="center", va="top", fontsize=6, color=INK2)
    ax.set_xlim(1, 30); ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("second $t$ of the epoch", labelpad=1); ax.set_ylabel("$\\|\\Delta_t\\|_F$", labelpad=1)
    ax.set_title("(a) over the epoch", fontsize=7, loc="left", pad=3)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.85, 1.1), columnspacing=0.8, handlelength=1.1,
              borderaxespad=0)
    nights = sorted(summary["change_level"]["stages"]["W"]["per_night"])
    night_mean = {n: np.mean([summary["change_level"]["stages"][name]["per_night"][n] for name in STAGES]) for n in nights}
    for k, name in enumerate(STAGES):
        levels = [summary["change_level"]["stages"][name]["per_night"][n] - night_mean[n] for n in nights]
        bx.scatter(np.arange(len(nights)) + (k - 2) * 0.11, levels, s=10, color=STAGE_COLORS[k], edgecolor="white", lw=0.3, zorder=3)
    bx.axhline(0, color=BASE, lw=0.7, zorder=1)
    bx.set_ylabel("settled $-$ night mean", labelpad=1)
    bx.set_xticks(range(len(nights)), [n[-2:] for n in nights]); bx.set_xlim(-0.5, len(nights) - 0.5)
    for boundary in np.arange(0.5, len(nights) - 1, 1.0):
        bx.axvline(boundary, color="#e1e0d9", lw=0.5)
    bx.set_xlabel("test night", labelpad=1)
    bx.set_title("(b) per test night", fontsize=7, loc="left", pad=3)
    for axis in (ax, bx):
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#e1e0d9", lw=0.5)
        axis.tick_params(length=2, pad=1.5)
    fig.savefig(OUT / "fig_stage_change.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "fig_stage_change.png", dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    night_name = sorted(set(subject))[0]
    in_night = np.flatnonzero(subject == night_name)
    in_night = in_night[np.argsort(run["onset"][in_night])]
    hours = np.arange(len(in_night)) * 30 / 3600
    level_of = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    trace = delta[in_night].mean(1)
    smooth = np.convolve(np.pad(trace, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.16, 1.15), sharex=True, gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]})
    fig.subplots_adjust(left=0.05, right=0.998)
    a.step(hours, [level_of[int(v)] for v in stage[in_night]], where="post", color=INK, lw=0.7)
    a.set_yticks(range(5), ["N3", "N2", "N1", "R", "W"]); a.set_ylim(-0.4, 4.4)
    b.plot(hours, smooth, color=POS, lw=0.9)
    b.set_ylabel("$\\|\\Delta_t\\|_F$", color=POS)
    for k in np.flatnonzero(stage[in_night] == 0):
        b.axvspan(hours[k], hours[k] + 30 / 3600, color=STAGE_COLORS[0], alpha=0.12, lw=0)
    b.set_xlabel(f"hours from the first scored epoch (test night {night_name[-2:]}; shaded: wake)")
    b.set_xlim(0, hours[-1])
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e1e0d9", lw=0.5)
        ax.tick_params(length=2, pad=1.5)
    fig.savefig(OUT / "fig_stage_night.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "fig_stage_night.png", dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    summary["night_trace"] = {"night": night_name, "corr_with_wake": float(np.corrcoef(trace, stage[in_night] == 0)[0, 1])}
    (NOTES / "stage_specificity.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
