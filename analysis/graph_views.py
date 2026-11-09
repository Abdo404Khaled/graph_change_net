"""Learned electrode graphs seen from four perspectives: 3-D, top (x-y), side (x-z), front (y-z).

Montage axes: x anterior (+, nose), y left (+), z up. Views are drawn the way an EEG reader
expects: top = seen from above, nose up, subject's left on the left; side = seen from the
left, nose to the left; front = seen from the front, subject's left on the viewer's right.
Edges: each electrode's top-k positive (row 1) and top-k negative (row 2) edges, each pair
once; line width and opacity follow the weight relative to the strongest edge of that sign.

Graphs (fold 0, classification weights; nothing is retrained):
  Exp 15 L^S, stable_only L^S, the fixed distance graph of distance_graph, the mean A_t of Exp 15
  over the 4,462 test epochs, and the stage-specific part of A_t (stage mean minus the mean over
  all test epochs) for W, N1, N2, N3 and R.
Outputs: one PNG per graph (2 signs x 4 views), a compact comparison figure (Exp 15 L^S vs
stable_only L^S vs distance graph, positive and negative edges), an interactive 3-D HTML with a
dropdown over all graphs (plots.graphs_3d), and a JSON with the edge counts and mean edge
lengths per graph. argv: output folder.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
import plots  # noqa: E402
from baselines.common.reference import scored_epochs  # noqa: E402
from config import load_run_config  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUN = REPO / "runs/exp15_final_fold0"
ABL = REPO / "ablations/results"
STAGES = ["W", "N1", "N2", "N3", "R"]
POS, NEG, INK, MUTED, BASE = "#e34948", "#2a78d6", "#0b0b0b", "#898781", "#c3c2b7"
K = 3
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASE,
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
cfg = load_run_config(RUN, {})
_, channels, coords = data.load_dataset(cfg)
coords = coords - coords.mean(0)
distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)

saved = np.load(RUN / "graphs.npz")
split = json.loads((RUN / "metrics.json").read_text())["split"]
table = scored_epochs(cfg, split["test"])
lookup = {(s, round(float(o), 3)): int(k) for s, t in table.items() for o, k in zip(t["onset"], t["stage"])}
stage = np.array([lookup[(s, round(float(o), 3))] for s, o in zip(saved["subject"], saved["onset"])])
epoch_graphs = saved["graph"].astype(np.float32)
mean_graph = epoch_graphs.mean(0)
graphs = {"Exp 15: stable graph L^S": np.load(RUN / "graphs.npz")["stable"],
          "stable_only: L^S (Delta_t = 0)": np.load(ABL / "stable_only_fold0/graphs.npz")["stable"],
          "distance_graph: fixed scalp-distance graph": np.load(ABL / "distance_graph_fold0/graphs.npz")["stable"],
          "Exp 15: mean A_t over the test epochs": mean_graph}
for k, name in enumerate(STAGES):
    graphs[f"Exp 15: stage-specific A_t, {name} (stage mean - overall mean, n = {(stage == k).sum()})"] = epoch_graphs[stage == k].mean(0) - mean_graph

VIEWS = [("3-D", None), ("top (x–y), nose up", lambda p: (-p[:, 1], p[:, 0])),
         ("side (x–z), seen from the left, nose left", lambda p: (-p[:, 0], p[:, 2])),
         ("front (y–z), seen from the front", lambda p: (p[:, 1], p[:, 2]))]


def top_k(matrix, positive):
    signed = 0.5 * (matrix + matrix.T)
    strength = signed if positive else -signed
    strength = np.where(np.eye(len(signed), dtype=bool), -np.inf, strength)
    pairs = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(np.argsort(-strength, 1)[:, :K])
             for j in row if strength[i, j] > 0}
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    rows, cols = map(np.asarray, zip(*sorted(pairs)))
    return rows, cols, signed[rows, cols]


def draw(ax, view, rows, cols, weights, colour):
    scale = np.abs(weights).max() + 1e-12 if len(weights) else 1.0
    order = np.argsort(np.abs(weights))
    if view is None:
        for e in order:
            i, j, r = rows[e], cols[e], abs(weights[e]) / scale
            ax.plot(*zip(coords[i], coords[j]), color=colour, lw=0.3 + 2.0 * r, alpha=0.2 + 0.65 * r)
        ax.scatter(*coords.T, s=6, color=INK, depthshade=False)
        ax.scatter(*coords[np.argmax(coords[:, 0])], s=40, marker="^", color=MUTED)
        ax.view_init(elev=22, azim=-125)
        ax.set_box_aspect(np.ptp(coords, 0), zoom=1.35); ax.set_axis_off()
        return
    x, y = view(coords)
    for e in order:
        i, j, r = rows[e], cols[e], abs(weights[e]) / scale
        ax.plot([x[i], x[j]], [y[i], y[j]], color=colour, lw=0.3 + 2.0 * r, alpha=0.2 + 0.65 * r)
    ax.scatter(x, y, s=6, color=INK, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def orientation_labels(ax, column):
    style = dict(transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="center", va="center")
    if column == 1:
        ax.text(-0.03, 0.5, "L", **style); ax.text(1.03, 0.5, "R", **style)
    elif column == 2:
        ax.text(-0.03, 0.5, "front", rotation=90, **style); ax.text(1.03, 0.5, "back", rotation=90, **style)
    elif column == 3:
        ax.text(-0.03, 0.5, "R", **style); ax.text(1.03, 0.5, "L", **style)


summary = {}
slug = lambda t: "".join(c if c.isalnum() else "_" for c in t.split(" (")[0]).strip("_").lower()  # noqa: E731
for title, matrix in graphs.items():
    fig = plt.figure(figsize=(17, 8))
    summary[title] = {}
    for row, (positive, colour, sign) in enumerate([(True, POS, "positive"), (False, NEG, "negative")]):
        rows, cols, weights = top_k(matrix, positive)
        lengths = distance[rows, cols] / distance[np.triu_indices(len(coords), 1)].mean() if len(rows) else np.zeros(0)
        summary[title][sign] = {"edges": int(len(rows)), "mean_length_relative": float(lengths.mean()) if len(rows) else None}
        for column, (label, view) in enumerate(VIEWS):
            ax = fig.add_subplot(2, 4, row * 4 + column + 1, projection="3d" if view is None else None)
            draw(ax, view, rows, cols, weights, colour)
            if view is not None:
                orientation_labels(ax, column)
            ax.set_title(f"3-D: {sign} edges ({len(rows)}),\nmean length {lengths.mean():.2f}× the mean distance" if column == 0 else label, fontsize=9)
    fig.suptitle(f"{title}: top {K} positive and top {K} negative edges per electrode (fold 0)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"views_{slug(title)}.png", dpi=150); plt.close(fig)

compare = ["Exp 15: stable graph L^S", "stable_only: L^S (Delta_t = 0)", "distance_graph: fixed scalp-distance graph"]
fig = plt.figure(figsize=(17, 12.5))
for r, title in enumerate(compare):
    for s, (positive, colour, sign) in enumerate([(True, POS, "positive"), (False, NEG, "negative")]):
        rows, cols, weights = top_k(graphs[title], positive)
        for column, (label, view) in enumerate([VIEWS[1], VIEWS[2]]):
            ax = fig.add_subplot(len(compare), 4, r * 4 + s * 2 + column + 1)
            if not len(rows):
                ax.text(0.5, 0.5, "no negative edges:\nthe fixed distance graph is non-negative", ha="center", va="center",
                        fontsize=9, color=MUTED, transform=ax.transAxes)
                ax.axis("off")
                continue
            draw(ax, view, rows, cols, weights, colour)
            orientation_labels(ax, [1, 2][column])
            ax.set_title(f"{title.split(':')[0]}, {sign}: {label.split(',')[0]}", fontsize=9)
fig.suptitle(f"Stable graphs compared: top {K} edges per electrode by sign, top and side views. "
             "The fixed distance graph has no negative edges.", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "views_compare_stable_graphs.png", dpi=150); plt.close(fig)

cfg.evaluation.top_k = K
plots.graphs_3d(cfg, graphs, channels, coords, OUT / "graphs_3d_all.html")
(OUT / "graph_views.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
