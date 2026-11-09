"""Stage-specific electrode graphs: Exp 15, the graph baselines, and the Pearson-correlation control.

Writes stage_summary.json and figures into the output folder.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
from baselines.common.reference import load_reference_config  # noqa: E402
from baselines.graph_tests import load_run, montage, stage_decoding, stages_of  # noqa: E402
from plots import scalp_layout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
STAGES = ["W", "N1", "N2", "N3", "R"]
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
settings = yaml.safe_load((REPO / "baselines/settings.yaml").read_text())
cfg = load_reference_config(REPO / settings["reference_run"])
names, coords = montage(cfg)
_xy = np.stack([-coords[:, 1], coords[:, 0]], 1); _xy = _xy - _xy.mean(0); _xy = 0.52 * _xy / np.linalg.norm(_xy, axis=1).max()
layout = {i: (float(_xy[i, 0]), float(_xy[i, 1])) for i in range(len(coords))}
rng = np.random.default_rng(0)

RUNS = {"Exp 15 (ours)": REPO / "runs/exp15_final_fold0/graphs.npz"}
for m, nice in [("graphsleepnet", "GraphSleepNet"), ("mstgcn", "MSTGCN"), ("productgraphsleepnet", "ProductGraphSleepNet")]:
    for c in ["ours", "native"]:
        RUNS[f"{nice} · {c}"] = REPO / f"baselines/results/{m}/{c}/graphs.npz"


def specificity(edges, stage, subject, perms=200):
    """Between-stage dispersion of stage means over its within-night label-shuffle null."""
    def stat(lbl):
        m = edges.mean(0)
        return sum(((edges[lbl == k].mean(0) - m) ** 2).sum() * (lbl == k).mean() for k in range(5) if (lbl == k).any())
    obs, null = stat(stage), []
    for _ in range(perms):
        sh = stage.copy()
        for s in np.unique(subject):
            idx = np.flatnonzero(subject == s); sh[idx] = rng.permutation(stage[idx])
        null.append(stat(sh))
    null = np.array(null)
    rms = np.sqrt(np.mean([((edges[stage == k].mean(0) - edges.mean(0)) ** 2).mean() for k in range(5)]))
    return {"ratio": float(obs / null.mean()), "p": float((1 + (null >= obs).sum()) / (perms + 1)),
            "relative_effect": float(rms / (np.abs(edges).mean() + 1e-12))}


summary, embeds = {}, {}
for name, path in RUNS.items():
    run = load_run(path, names)
    stage = stages_of(cfg, run)
    iu = np.triu_indices(run["graph"].shape[1], 1)
    edges = run["graph"][:, iu[0], iu[1]]
    if np.allclose(edges, edges[:1]):
        continue
    summary[name] = {"nodes": int(run["graph"].shape[1]), "decoding": stage_decoding(run, stage, 50, 0),
                     "specificity": specificity(edges, stage, run["subject"])}
    sel = rng.choice(len(edges), size=min(2500, len(edges)), replace=False)
    z = PCA(30, random_state=0).fit_transform((edges[sel] - edges.mean(0)) / (edges.std(0) + 1e-9))
    embeds[name] = (TSNE(2, random_state=0, init="pca", perplexity=40).fit_transform(z), stage[sel])
    print(name, summary[name], flush=True)

channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
signature = data.preprocessing_signature(cfg, channels)
exp15 = load_run(RUNS["Exp 15 (ours)"], names)
full, seg = [], []
for subject in np.unique(exp15["subject"]):
    meta = np.load(REPO / cfg.paths.cache_dir / f"{subject}_{signature}_epochs_meta.npz")
    X = np.load(REPO / cfg.paths.cache_dir / f"{subject}_{signature}_epochs.npy", mmap_mode="r")
    row = {round(float(o), 3): i for i, o in enumerate(meta["onset"])}
    for o in exp15["onset"][exp15["subject"] == subject]:
        x = np.asarray(X[row[round(float(o), 3)]], dtype=np.float32)
        full.append(np.corrcoef(x))
        s = x.reshape(x.shape[0], 30, -1).transpose(1, 0, 2)
        s = (s - s.mean(-1, keepdims=True)) / (s.std(-1, keepdims=True) + 1e-6)
        seg.append(np.einsum("tcl,tdl->cd", s, s) / (30 * s.shape[-1]))
for label, mats in [("Pearson, 30 s epoch", full), ("Pearson, mean of 1 s segments", seg)]:
    run = {**exp15, "graph": np.nan_to_num(np.array(mats, dtype=np.float32))}
    stage = stages_of(cfg, run)
    iu = np.triu_indices(83, 1)
    summary[label] = {"nodes": 83, "decoding": stage_decoding(run, stage, 50, 0),
                      "specificity": specificity(run["graph"][:, iu[0], iu[1]], stage, run["subject"])}
    print(label, summary[label], flush=True)
(OUT / "stage_summary.json").write_text(json.dumps(summary, indent=1))

g = exp15["graph"]; st = stages_of(cfg, exp15); subj = exp15["subject"]
iu = np.triu_indices(83, 1); E = g[:, iu[0], iu[1]]
diff = np.array([E[st == k].mean(0) - E.mean(0) for k in range(5)])
null = []
for _ in range(200):
    sh = st.copy()
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s); sh[idx] = rng.permutation(st[idx])
    null.append(np.array([E[sh == k].mean(0) - E.mean(0) for k in range(5)]))
null = np.abs(np.array(null))
pvals = (1 + (null >= np.abs(diff)[None]).sum(0)) / 201
fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
xy = np.array([layout[i] for i in range(83)])
for k, ax in enumerate(axes):
    ax.add_patch(plt.Circle((0, 0), 0.56, fill=False, color="#888", lw=1)); ax.plot([-.05, 0, .05], [.55, .62, .55], color="#888", lw=1)
    sig = np.flatnonzero(pvals[k] < 0.005)
    top = sig[np.argsort(-np.abs(diff[k, sig]))[:60]]
    vmax = np.abs(diff[k, top]).max() if len(top) else 1
    for e in top[::-1]:
        i, j = iu[0][e], iu[1][e]; v = diff[k, e]
        ax.plot(*zip(xy[i], xy[j]), color="#d62728" if v > 0 else "#1f4fd1", lw=0.6 + 3.2 * abs(v) / vmax, alpha=0.85)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, color="#222", zorder=3)
    ax.set_title(f"{STAGES[k]}  ({(st == k).sum()} epochs, {len(sig)} edges p<0.005)", fontsize=11)
    ax.set_xlim(-.62, .62); ax.set_ylim(-.62, .68); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Exp 15 electrode graph: edges stronger (red) or weaker (blue) in each stage than on average; top 60 significant edges", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "exp15_stage_edges.png", dpi=130); plt.close(fig)

import mne  # noqa: E402
strength = np.abs(g).sum(-1)
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
dd = np.array([strength[st == k].mean(0) - strength.mean(0) for k in range(5)])
lim = np.abs(dd).max()
for k, ax in enumerate(axes):
    im, _ = mne.viz.plot_topomap(dd[k], xy, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim), sphere=0.56, contours=4)
    ax.set_title(STAGES[k], fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.8, label="node strength vs mean")
fig.suptitle("Exp 15: how strongly each electrode is connected in each stage, relative to its average", fontsize=12)
fig.savefig(OUT / "exp15_stage_strength.png", dpi=130, bbox_inches="tight"); plt.close(fig)

def region(n):
    for pre, r in [("FP", "prefrontal"), ("AF", "prefrontal"), ("SO", "periocular"), ("ZY", "periocular"), ("FT", "temporal"), ("TP", "temporal"),
                   ("T", "temporal"), ("F", "frontal"), ("C", "central"), ("P", "parietal"), ("O", "occipital")]:
        if n.startswith(pre):
            return r
    return "other"
def side(n):
    d = "".join(ch for ch in n if ch.isdigit())
    return "z" if not d else ("L" if int(d) % 2 else "R")
labels = [f"{region(n)}-{side(n)}" for n in names]
order = sorted(set(labels), key=lambda s: (["prefrontal", "frontal", "central", "parietal", "occipital", "temporal", "periocular", "other"].index(s.split("-")[0]), s))
idx = {l: [i for i, x in enumerate(labels) if x == l] for l in order}
def blocks(mat):
    return np.array([[mat[np.ix_(idx[a], idx[b])].mean() for b in order] for a in order])
mean_g = g.mean(0)
bd = [blocks(g[st == k].mean(0) - mean_g) for k in range(5)]
lim = max(np.abs(b).max() for b in bd)
fig, axes = plt.subplots(1, 5, figsize=(24, 5.4))
for k, ax in enumerate(axes):
    im = ax.imshow(bd[k], cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7); ax.set_yticklabels(order if k == 0 else [], fontsize=7)
    ax.set_title(STAGES[k], fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.7, label="mean edge weight vs overall mean")
fig.suptitle("Exp 15: region-to-region edge weight in each stage minus the average (L = left, R = right, z = midline)", fontsize=12)
fig.savefig(OUT / "exp15_stage_blocks.png", dpi=130, bbox_inches="tight"); plt.close(fig)

models = [m for m in ["Exp 15 (ours)", "GraphSleepNet · ours", "MSTGCN · ours", "ProductGraphSleepNet · ours"] if m in embeds]
fig, axes = plt.subplots(len(models), 5, figsize=(16, 3.2 * len(models)))
for r, m in enumerate(models):
    emb, lab = embeds[m]
    for k in range(5):
        ax = axes[r, k]
        ax.scatter(emb[:, 0], emb[:, 1], s=2, color="#cfd3da")
        ax.scatter(emb[lab == k, 0], emb[lab == k, 1], s=3, color=STAGE_COLORS[k])
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(STAGES[k], fontsize=12)
        if k == 0: ax.set_ylabel(f"{m}\nκ from graph {summary[m]['decoding']['kappa']:.2f}", fontsize=10)
fig.suptitle("Each epoch's electrode graph embedded in 2-D (t-SNE); each column highlights one stage", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "stage_tsne.png", dpi=120); plt.close(fig)

z = np.load(RUNS["Exp 15 (ours)"]); dn = z["delta_norm"].astype(np.float32).mean(1)
sel = exp15["subject"] == "EPCTL06"; o = exp15["onset"][sel]; order_t = np.argsort(o)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
hrs = o[order_t] / 3600
a1.plot(hrs, dn[sel][order_t], color="#2a78d6", lw=0.8); a1.set_ylabel("‖Δ‖ (mean of 30 segments)")
depth = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
a2.step(hrs, [depth[s] for s in st[sel][order_t]], where="post", color="#333", lw=1)
a2.set_yticks([0, 1, 2, 3, 4]); a2.set_yticklabels(["N3", "N2", "N1", "REM", "W"]); a2.set_xlabel("hours from recording start")
fig.suptitle("Exp 15, test night EPCTL06: size of the change term against the hypnogram", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "exp15_delta_night.png", dpi=130); plt.close(fig)

dn_t = z["delta_norm"].astype(np.float32)
fig, ax = plt.subplots(figsize=(7, 3.6))
for k in range(5):
    m = dn_t[st == k].mean(0); ax.plot(np.arange(1, 31), m, color=STAGE_COLORS[k], lw=2, label=STAGES[k])
ax.set_xlabel("segment (1 s) within the epoch"); ax.set_ylabel("‖Δ_t‖"); ax.legend(ncol=5, frameon=False, fontsize=9)
ax.set_title("Exp 15: size of Δ_t through the epoch, by stage")
fig.tight_layout(); fig.savefig(OUT / "exp15_delta_profile.png", dpi=130); plt.close(fig)
print("figures written", flush=True)
