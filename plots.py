from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from config import Config

plt.switch_backend("Agg")


def _save(fig, path: Path) -> None:
    """Write a figure to disk and release it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : Path
        Output file.
    """
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


TASK_COLOURS = {"reconstruction": "tab:blue", "spectral": "tab:orange", "future": "tab:green"}


def stage1_panel(ax, cfg: Config, pretrain_history: list) -> None:
    """Draw the stage-1 curves: every task on training and validation epochs, or the old pair.

    Rows with curves (``evaluation.curve_batches``) hold every stage-1 loss
    relative to its trivial baseline, scored the same way on a fixed
    training subset and on the validation epochs, so the two lines of a task
    are comparable and 1 means "no better than trivial". Older rows are
    ``(epoch, running training loss, validation loss)``, drawn as before.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    cfg : Config
        Run settings.
    pretrain_history : list
        History rows from :func:`train.pretrain`; may be empty.
    """
    if not pretrain_history:
        ax.text(0.5, 0.5, "No stage 1 in this run", ha="center", va="center")
        ax.set_axis_off()
        return
    if not isinstance(pretrain_history[0], dict):
        epochs, train_loss, val_loss = zip(*pretrain_history)
        ax.plot(epochs, train_loss, label="Train (running, augmented)")
        ax.plot(epochs, val_loss, label="Validation")
        ax.set(title="Stage 1 - Reconstruction loss", xlabel="Epoch", ylabel="Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        return
    epochs = [row["epoch"] for row in pretrain_history]
    for task, colour in TASK_COLOURS.items():
        if task not in pretrain_history[0]["val_curves"]:
            continue
        ax.plot(epochs, [row["train_curves"][task] for row in pretrain_history], color=colour,
                marker="o", markersize=3, label=f"{task} - train")
        ax.plot(epochs, [row["val_curves"][task] for row in pretrain_history], color=colour,
                marker="o", markersize=3, linestyle="--", label=f"{task} - validation")
    skipped = sum(row["skipped"] for row in pretrain_history)
    steps = sum(row["steps"] for row in pretrain_history)
    ax.axhline(1.0, color="0.5", linewidth=0.8)
    ax.set(title=f"Stage 1 - every task, relative to its trivial baseline\n"
                 f"(eval mode, same masks; non-finite steps skipped: {skipped} of {steps})",
           xlabel="Epoch", ylabel="Loss / trivial loss")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)


def stage2_panels(ax_scores, ax_loss, cfg: Config, history: list[dict]) -> None:
    """Draw the stage-2 curves: scores and loss, training against validation when available.

    With curves, both panels compare a fixed training subset and the
    validation epochs scored by one function (:func:`evaluate.curve_scores`):
    macro-F1 and kappa on the left, the plain cross entropy on the right.
    Without, the left panel shows the validation scores and the right one
    the running training loss, which is taken with dropout, augmentation and
    a weighted focal loss and is not comparable with a validation score.

    Parameters
    ----------
    ax_scores, ax_loss : matplotlib.axes.Axes
        Target axes.
    cfg : Config
        Run settings.
    history : list of dict
        Rows from :func:`train.classify`; may be empty.
    """
    if not history:
        for ax in (ax_scores, ax_loss):
            ax.set_axis_off()
        return
    epochs = np.arange(1, len(history) + 1)
    curves = "train_curves" in history[0]
    for key, label, colour in [("macro_f1", "Macro-F1", "tab:blue"),
                               ("kappa", "Kappa", "tab:orange")]:
        if curves:
            ax_scores.plot(epochs, [row["train_curves"][key] for row in history], color=colour,
                           label=f"{label} - train")
        ax_scores.plot(epochs, [row[key] for row in history], color=colour,
                       linestyle="--" if curves else "-", label=f"{label} - validation")
    if curves:
        ax_loss.plot(epochs, [row["train_curves"]["cross_entropy"] for row in history],
                     label="Train")
        ax_loss.plot(epochs, [row["val_curves"]["cross_entropy"] for row in history],
                     linestyle="--", label="Validation")
        ax_loss.set(title="Stage 2 - cross entropy (eval mode, unweighted)", ylabel="Cross entropy")
    else:
        ax_loss.plot(epochs, [row["train_loss"] for row in history], label="Train (running)")
        ax_loss.set(title="Stage 2 - running training loss (not comparable with validation)",
                    ylabel="Loss")
    for ax in (ax_scores, ax_loss):
        if cfg.finetune.freeze_epochs < len(history):
            ax.axvline(cfg.finetune.freeze_epochs + 0.5, color="0.5", linestyle=":",
                       label="Backbone unfrozen")
        ax.set(xlabel="Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    ax_scores.set(title="Stage 2 - scores" + (" (train vs validation)" if curves else
                                              " (validation)"), ylabel="Score", ylim=(0, 1))


def summary(cfg: Config, pretrain_history: list, history: list[dict], test_metrics: dict,
            true: np.ndarray, predicted: np.ndarray, counts: dict[str, tuple[int, int]],
            path: Path) -> None:
    """Training curves, per-stage F1, confusion matrices, class distribution and split sizes.

    Parameters
    ----------
    cfg : Config
        Run settings.
    pretrain_history : list
        Rows from :func:`train.pretrain` (tuples, or dictionaries with
        curves); may be empty.
    history : list of dict
        Rows from :func:`train.classify`; may be empty.
    test_metrics : dict
        Output of :func:`evaluate.scores` on the test split.
    true, predicted : np.ndarray
        Test stage indices.
    counts : dict
        ``split -> (subjects, items)`` from :func:`run.split_counts`.
    path : Path
        Output file.
    """
    stages = cfg.data.stages
    cm = confusion_matrix(true, predicted, labels=range(len(stages)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(2, 4, figsize=(25, 11))

    stage1_panel(axes[0, 0], cfg, pretrain_history)
    stage2_panels(axes[0, 1], axes[0, 2], cfg, history)

    axes[0, 3].bar(stages, [test_metrics[f"f1_{stage}"] for stage in stages])
    for index, stage in enumerate(stages):
        axes[0, 3].text(index, test_metrics[f"f1_{stage}"] + 0.02,
                        f"{test_metrics[f'f1_{stage}']:.2f}", ha="center")
    axes[0, 3].set(title="Test F1 per sleep stage", xlabel="Sleep stage", ylabel="F1",
                   ylim=(0, 1.05))
    axes[0, 3].grid(axis="y", alpha=0.25)

    for ax, matrix, title, fmt in [(axes[1, 0], cm, "counts", "d"),
                                   (axes[1, 1], cm_norm, "row-normalised", ".2f")]:
        image = ax.imshow(matrix) if fmt == "d" else ax.imshow(matrix, vmin=0, vmax=1)
        ax.set(title=f"Test confusion matrix - {title}", xlabel="Predicted", ylabel="True")
        ax.set_xticks(range(len(stages)), stages)
        ax.set_yticks(range(len(stages)), stages)
        for i in range(len(stages)):
            for j in range(len(stages)):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046)

    x, width = np.arange(len(stages)), 0.38
    axes[1, 2].bar(x - width / 2, np.bincount(true, minlength=len(stages)), width, label="True")
    axes[1, 2].bar(x + width / 2, np.bincount(predicted, minlength=len(stages)), width,
                   label="Predicted")
    axes[1, 2].set_xticks(x, stages)
    axes[1, 2].set(title="Test class distribution", xlabel="Sleep stage", ylabel="Epochs")
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", alpha=0.25)

    lines = ["Data (subjects, epochs used)", ""]
    for name, (subjects, items) in counts.items():
        lines.append(f"{name:<9}{subjects:>3} subjects  {items:>7,} epochs")
    lines += ["", f"stage 1 batch {cfg.loader.batch_size}, stage 2 batch "
                  f"{cfg.finetune.batch_size or cfg.loader.batch_size}",
              f"segments {cfg.data.n_segments} x {cfg.data.window_s / cfg.data.n_segments:g} s, "
              f"context {cfg.data.context_epochs} epoch(s)",
              f"decoder {cfg.model.decoder['name']}"]
    axes[1, 3].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
                    fontsize=11, transform=axes[1, 3].transAxes)
    axes[1, 3].set_axis_off()

    fig.suptitle(f"{cfg.experiment}\nKappa={test_metrics['kappa']:.3f} | "
                 f"Macro-F1={test_metrics['macro_f1']:.3f} | "
                 f"Accuracy={test_metrics['accuracy']:.3f}", fontsize=15)
    plt.tight_layout()
    _save(fig, path)


@torch.no_grad()
def reconstruction_examples(cfg: Config, model, dataset, path: Path, seed: int = 0) -> None:
    """One whole-electrode reconstruction per sleep stage.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : GraphChangeNet
        Network with its stage 1 weights.
    dataset : WindowDataset
        Windows to draw the examples from.
    path : Path
        Output file.
    seed : int
        Seed of the window and electrode choice.
    """
    model.eval()
    device, data = cfg.device, cfg.data
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(data.stages), 1, figsize=(15, 12), sharex=True)

    for stage_index, stage in enumerate(data.stages):
        ax = axes[stage_index]
        indices = np.where(dataset.y == stage_index)[0]
        if len(indices) == 0:
            ax.text(0.5, 0.5, f"No {stage} sample", ha="center", va="center")
            ax.set_axis_off()
            continue

        x, *_ = dataset[rng.choice(indices)]
        x = x.unsqueeze(0).to(device)
        electrode = rng.integers(0, x.shape[1])
        mask = torch.zeros(1, data.n_segments, x.shape[1], dtype=torch.bool, device=device)
        mask[:, :, electrode] = True

        rebuilt_segments, _ = model.reconstruct(x, mask)
        original_segments = model.segments(x)
        original = original_segments[0, :, electrode].reshape(-1).cpu().numpy()
        rebuilt = rebuilt_segments[0, :, electrode].reshape(-1).cpu().numpy()
        loss = F.smooth_l1_loss(rebuilt_segments[0, :, electrode],
                                original_segments[0, :, electrode]).item()
        corr = np.corrcoef(original, rebuilt)[0, 1]

        time = np.arange(len(original)) / data.sfreq
        ax.plot(time, original, label="Original", linewidth=1.4)
        ax.plot(time, rebuilt, label="Reconstructed", linewidth=1.4)
        for segment in range(1, data.n_segments):
            ax.axvline(segment * data.segment_len / data.sfreq, linestyle="--", alpha=0.25)
        ax.set_title(f"{stage} | Electrode {electrode} | Loss={loss:.4f} | Corr={corr:.3f}")
        ax.set_ylabel("Scaled EEG")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend()
    fig.suptitle("Whole-Electrode Reconstruction - One Random Example per Sleep Stage",
                 fontsize=15)
    plt.tight_layout()
    _save(fig, path)


def scalp_layout(coords: np.ndarray) -> dict[int, tuple[float, float]]:
    """Project electrode positions onto a unit head for 2-D drawings.

    The montage's x axis points to the nose and its y axis to the left ear,
    while :func:`draw_scalp_graph` draws the nose at the top. The drawing's
    horizontal axis is therefore ``-y`` (the subject's right on the right) and
    its vertical axis ``x``; taking ``(x, y)`` directly would turn every scalp
    figure by 90 degrees.

    Parameters
    ----------
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.

    Returns
    -------
    dict
        ``electrode index -> (x, y)`` inside a circle of radius 0.52.
    """
    xy = np.stack([-coords[:, 1], coords[:, 0]], axis=1)
    xy -= xy.mean(axis=0)
    xy = 0.52 * xy / np.max(np.linalg.norm(xy, axis=1))
    return {i: (float(xy[i, 0]), float(xy[i, 1])) for i in range(len(coords))}


def draw_scalp_graph(ax, matrix: np.ndarray, title: str, reference_values: np.ndarray,
                     layout: dict) -> None:
    """Draw one signed graph on a head outline, stronger edges thicker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    matrix : np.ndarray
        Signed graph ``[C, C]``.
    title : str
        Panel title.
    reference_values : np.ndarray
        Absolute weights that set the 90 / 95 / 98 / 99.5 percentile bands, so
        graphs of one family share a scale.
    layout : dict
        Output of :func:`scalp_layout`.
    """
    n = matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from((i, j, float(matrix[i, j])) for i, j in zip(rows, cols))
    p90, p95, p98, p995 = np.percentile(reference_values, [90, 95, 98, 99.5])

    ax.add_patch(plt.Circle((0, 0), 0.55, fill=False, color="0.45", linewidth=1))
    ax.plot([-0.04, 0, 0.04], [0.55, 0.60, 0.55], color="0.45", linewidth=1)
    ax.plot([-0.55, -0.58, -0.58, -0.55], [0.10, 0.06, -0.06, -0.10], color="0.45", linewidth=1)
    ax.plot([0.55, 0.58, 0.58, 0.55], [0.10, 0.06, -0.06, -0.10], color="0.45", linewidth=1)

    edges = list(graph.edges(data=True))

    def draw(selected, width, alpha):
        """Draw a set of edges, red for positive and blue for negative.

        Parameters
        ----------
        selected : list
            ``(u, v, attributes)`` edges.
        width : float
            Line width.
        alpha : float
            Line opacity.
        """
        if selected:
            nx.draw_networkx_edges(graph, layout, ax=ax, edgelist=[(u, v) for u, v, _ in selected],
                                   edge_color=["red" if d["weight"] >= 0 else "blue"
                                               for _, _, d in selected],
                                   width=width, alpha=alpha)

    draw(edges, 0.12, 0.015)
    for threshold, width, alpha in [(p90, 0.35, 0.08), (p95, 0.70, 0.18),
                                    (p98, 1.30, 0.40), (p995, 2.50, 0.85)]:
        draw([e for e in edges if abs(e[2]["weight"]) >= threshold], width, alpha)

    nx.draw_networkx_nodes(graph, layout, ax=ax, node_size=11, node_color="0.25")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.64, 0.64)
    ax.set_ylim(-0.64, 0.64)
    ax.set_aspect("equal")
    ax.axis("off")


def plotted_segments(cfg: Config) -> list[int]:
    """Indices of the segments whose change is drawn in the graph figures.

    Parameters
    ----------
    cfg : Config
        Run settings (``data.n_segments``, ``evaluation.plot_segments``).

    Returns
    -------
    list of int
        Every segment, or ``plot_segments`` of them evenly spaced over the window.
    """
    n_segments, wanted = cfg.data.n_segments, cfg.evaluation.plot_segments
    if wanted <= 0 or wanted >= n_segments:
        return list(range(n_segments))
    return sorted({int(t) for t in np.linspace(0, n_segments - 1, wanted).round()})


def scalp_graphs(cfg: Config, stable: np.ndarray, deltas: np.ndarray, coords: np.ndarray,
                 path: Path, window: int = 0) -> None:
    """``L^S``, the mean change and ``L^S + Delta`` of one window, and its ``Delta_t``.

    Parameters
    ----------
    cfg : Config
        Run settings.
    stable : np.ndarray
        Stable graph ``[C, C]``.
    deltas : np.ndarray
        Changes ``[N, T, C, C]``.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    path : Path
        Output file.
    window : int
        Window whose changes are drawn.
    """
    data = cfg.data
    layout = scalp_layout(coords)
    mean_delta = deltas[window].mean(0)
    combined = stable + mean_delta
    upper = np.triu_indices_from(stable, k=1)
    stable_reference = np.concatenate([np.abs(stable[upper]), np.abs(combined[upper])])
    segments = plotted_segments(cfg)
    delta_reference = np.concatenate([np.abs(mean_delta[upper]),
                                      *[np.abs(deltas[window, t][upper]) for t in segments]])
    segment_ms = 1000 * data.window_s / data.n_segments
    columns = max(len(segments), 3)

    fig, axes = plt.subplots(2, columns, figsize=(3 * columns, 6.2))
    draw_scalp_graph(axes[0, 0], stable, r"Stable graph $L^S$", stable_reference, layout)
    draw_scalp_graph(axes[0, 1], mean_delta, r"Mean dynamic change $\Delta$", delta_reference,
                     layout)
    draw_scalp_graph(axes[0, 2], combined, r"$L_t = L^S + \Delta_t$", stable_reference, layout)
    for ax in axes[0, 3:]:
        ax.axis("off")
    for column, t in enumerate(segments):
        span = f"{segment_ms * t:.0f}-{segment_ms * (t + 1):.0f} ms"
        draw_scalp_graph(axes[1, column], deltas[window, t],
                         rf"$\Delta_{{t={t + 1}}}$   {span}", delta_reference, layout)
    for ax in axes[1, len(segments):]:
        ax.axis("off")

    legend = [mlines.Line2D([], [], color="red", lw=2, label="Positive"),
              mlines.Line2D([], [], color="blue", lw=2, label="Negative"),
              mlines.Line2D([], [], color="gray", lw=0.6, alpha=.15, label="< 90th percentile"),
              mlines.Line2D([], [], color="gray", lw=.7, label="Top 10%"),
              mlines.Line2D([], [], color="gray", lw=2.0, alpha=.7, label="Top 2%"),
              mlines.Line2D([], [], color="gray", lw=3.0, label="Top 0.5%")]
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, path)


def top_k_edges(matrix: np.ndarray, k: int, positive: bool):
    """The ``k`` strongest edges of one sign at every electrode.

    For each electrode, its ``k`` largest positive (or most negative) edges
    are selected; the union over electrodes is returned with each pair once,
    so an edge chosen by both of its electrodes is drawn a single time. An
    electrode with fewer than ``k`` edges of that sign keeps all of them.
    The graph is symmetrised first, so a pair has one weight.

    Parameters
    ----------
    matrix : np.ndarray
        Signed graph ``[C, C]``.
    k : int
        Edges of the chosen sign kept per electrode.
    positive : bool
        Select positive edges, otherwise negative ones.

    Returns
    -------
    rows, cols : np.ndarray
        Electrode indices of the kept pairs, ``rows < cols``.
    weights : np.ndarray
        Their signed weights.
    """
    signed = 0.5 * (matrix + matrix.T)
    strength = signed if positive else -signed
    strength = np.where(np.eye(len(strength), dtype=bool), -np.inf, strength)
    order = np.argsort(-strength, axis=1)[:, :k]
    pairs = set()
    for node, neighbours in enumerate(order):
        for neighbour in neighbours:
            if strength[node, neighbour] > 0:
                pairs.add((min(node, neighbour), max(node, neighbour)))
    if not pairs:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0, dtype=signed.dtype)
    rows, cols = map(np.asarray, zip(*sorted(pairs)))
    return rows, cols, signed[rows, cols]


def stage_graphs(cfg: Config, stable: np.ndarray, deltas: np.ndarray, stages: np.ndarray,
                 window: int = 0) -> dict[str, np.ndarray]:
    """The graphs worth inspecting: ``L^S``, the mean graph per stage and one window's changes.

    Parameters
    ----------
    cfg : Config
        Run settings.
    stable : np.ndarray
        Stable graph ``[C, C]``.
    deltas : np.ndarray
        Changes ``[N, T, C, C]``.
    stages : np.ndarray
        Stage of every window ``[N]``.
    window : int
        Window whose changes are included.

    Returns
    -------
    dict
        ``title -> graph [C, C]``.
    """
    data = cfg.data
    graphs = {"L^S (stable)": stable,
              "L^S + mean Delta, all test windows": stable + deltas.mean((0, 1))}
    for index, stage in enumerate(data.stages):
        if (stages == index).any():
            mean_change = deltas[stages == index].mean((0, 1))
            graphs[f"L^S + mean Delta, stage {stage}"] = stable + mean_change
    segment_ms = 1000 * data.window_s / data.n_segments
    for t in plotted_segments(cfg):
        graphs[f"Delta_t={t + 1} ({segment_ms * t:.0f}-{segment_ms * (t + 1):.0f} ms), "
               f"test window {window} ({data.stages[stages[window]]})"] = deltas[window, t]
    return graphs


def graphs_3d(cfg: Config, graphs: dict[str, np.ndarray], channels: list[str],
              coords: np.ndarray, path: Path) -> None:
    """Interactive plotly view of signed graphs on the real electrode positions.

    A dropdown switches between graphs; positive and negative edges are
    separate legend entries so either sign can be hidden.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.evaluation.top_k``).
    graphs : dict
        ``title -> graph [C, C]``.
    channels : list of str
        Electrode names.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    path : Path
        Output HTML file.
    """
    import plotly.graph_objects as go

    k = cfg.evaluation.top_k
    selection = f"top {k} positive + top {k} negative edges per electrode"

    def edge_trace(matrix: np.ndarray, positive: bool):
        """One line trace with the top-k edges of one sign at every electrode.

        Parameters
        ----------
        matrix : np.ndarray
            Signed graph ``[C, C]``.
        positive : bool
            Draw the positive edges, otherwise the negative ones.

        Returns
        -------
        plotly.graph_objects.Scatter3d
            The trace, hidden until selected.
        """
        rows, cols, weights = top_k_edges(matrix, k, positive)
        scale = np.abs(weights).max() + 1e-12 if len(weights) else 1.0
        xs, ys, zs, colour, hover = [], [], [], [], []
        for i, j, weight in zip(rows, cols, weights):
            label = f"{channels[i]} - {channels[j]}: {weight:+.3f}"
            for node in (i, j):
                xs.append(coords[node, 0])
                ys.append(coords[node, 1])
                zs.append(coords[node, 2])
                colour.append(abs(weight) / scale)
                hover.append(label)
            for values, gap in [(xs, None), (ys, None), (zs, None), (colour, 0.0), (hover, None)]:
                values.append(gap)
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", hovertext=hover, hoverinfo="text",
            name=f"{'positive' if positive else 'negative'} ({len(weights)} edges)",
            legendgroup="positive" if positive else "negative",
            line=dict(width=5, color=colour, cmin=0, cmax=1,
                      colorscale="Reds" if positive else "Blues"),
            visible=False)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], alphahull=0,
                            opacity=0.07, color="lightgray", hoverinfo="skip", name="scalp"))
    fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                               mode="markers+text", text=channels, textfont=dict(size=8),
                               hoverinfo="text", marker=dict(size=3, color="dimgray"),
                               name="electrodes"))
    for matrix in graphs.values():
        fig.add_trace(edge_trace(matrix, True))
        fig.add_trace(edge_trace(matrix, False))

    fixed = 2
    buttons = []
    for k, name in enumerate(graphs):
        visible = [True] * fixed + [False] * (2 * len(graphs))
        visible[fixed + 2 * k] = visible[fixed + 2 * k + 1] = True
        buttons.append(dict(label=name, method="update",
                            args=[{"visible": visible},
                                  {"title": f"{name} - {selection}"}]))
    for trace in fig.data[fixed:fixed + 2]:
        trace.visible = True

    fig.update_layout(
        title=f"{next(iter(graphs))} - {selection}",
        updatemenus=[dict(buttons=buttons, x=0.0, y=1.08, xanchor="left", showactive=True)],
        scene=dict(aspectmode="data", xaxis_title="x (anterior)", yaxis_title="y (left)",
                   zaxis_title="z (up)"),
        legend=dict(title="click to hide a sign"), height=750, margin=dict(l=0, r=0, t=80, b=0))
    fig.write_html(path, include_plotlyjs="cdn")


def graph_projections(cfg: Config, matrix: np.ndarray, title: str, coords: np.ndarray,
                      path: Path) -> None:
    """Static views of one signed graph, positive and negative edges in separate rows.

    Each row shows a 3-D view plus the top, side and front projections of the
    ``top_k`` strongest edges of one sign at every electrode. Line width and
    opacity follow the weight relative to the strongest edge of that sign.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.evaluation.top_k``).
    matrix : np.ndarray
        Signed graph ``[C, C]``.
    title : str
        Figure title.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    path : Path
        Output file.
    """
    k = cfg.evaluation.top_k
    views = [("3-D", None), ("top (x-y)", (0, 1)), ("side (x-z)", (0, 2)), ("front (y-z)", (1, 2))]
    fig = plt.figure(figsize=(20, 10))
    for row, (positive, colour) in enumerate([(True, "red"), (False, "blue")]):
        rows, cols, weights = top_k_edges(matrix, k, positive)
        scale = np.abs(weights).max() + 1e-12 if len(weights) else 1.0
        sign = "positive" if positive else "negative"
        for column, (label, axes_pair) in enumerate(views):
            panel = row * len(views) + column + 1
            if axes_pair is None:
                ax = fig.add_subplot(2, len(views), panel, projection="3d")
                points = coords
            else:
                ax = fig.add_subplot(2, len(views), panel)
                points = coords[:, axes_pair]
            for i, j, weight in zip(rows, cols, weights):
                relative = abs(weight) / scale
                ax.plot(*zip(points[i], points[j]), color=colour,
                        linewidth=0.3 + 2.2 * relative, alpha=0.25 + 0.6 * relative)
            ax.scatter(*points.T, s=8, color="0.25")
            ax.set_title(f"{sign} ({len(weights)} edges) - {label}")
            if axes_pair is not None:
                ax.set_aspect("equal")
                ax.set_xlabel("xyz"[axes_pair[0]])
                ax.set_ylabel("xyz"[axes_pair[1]])
    fig.suptitle(f"{title} - top {k} positive and top {k} negative edges per electrode")
    plt.tight_layout()
    _save(fig, path)


def segment_graphs(cfg: Config, graphs: dict, path: Path) -> None:
    """The learned segment graph: edge signs, importance and importance by lag, per stage.

    One row per group (all test windows, then each stage). Columns: the mean
    probability that an edge is positive (discrete edges) or the mean signed
    edge weight (continuous edges), red positive and blue negative;
    the importance of every earlier segment on the positive and on the
    negative side, relative to equal attention (1 = an even share); and the
    mean relative importance of the segment ``k`` steps back, for both signs.

    Parameters
    ----------
    cfg : Config
        Run settings (segment length).
    graphs : dict
        Output of :func:`evaluate.collect_segment_graphs`.
    path : Path
        Output file.
    """
    groups = graphs["groups"]
    segment_ms = 1000 * cfg.data.window_s / cfg.data.n_segments
    sign_low, sign_high = float(graphs["edge_low"]), 1.0
    if sign_low < 0 and np.isfinite(graphs["sign"]).any():
        sign_high = max(float(np.nanpercentile(np.abs(graphs["sign"]), 99)), 1e-6)
        sign_low = -sign_high
    importance = np.concatenate([graphs["positive"], graphs["negative"]])
    top = np.nanpercentile(importance, 99) if np.isfinite(importance).any() else 1.0
    fig, axes = plt.subplots(len(groups), 4, figsize=(17, 3.6 * len(groups)), squeeze=False)
    for row, name in enumerate(groups):
        title = f"{name} ({int(graphs['counts'][row])} windows)"
        panels = [(graphs["sign"][row], "coolwarm", sign_low, sign_high,
                   str(graphs["edge_label"])),
                  (graphs["positive"][row], "viridis", 0.0, top, "importance, + edges"),
                  (graphs["negative"][row], "viridis", 0.0, top, "importance, - edges")]
        for column, (matrix, cmap, low, high, label) in enumerate(panels):
            ax = axes[row, column]
            image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, origin="upper")
            ax.set_title(f"{title}\n{label}", fontsize=9)
            ax.set_xlabel("earlier segment s")
            ax.set_ylabel("segment t")
            fig.colorbar(image, ax=ax, fraction=0.046)
        ax = axes[row, 3]
        lags = np.arange(1, matrix.shape[0])
        for key, colour in (("positive", "tab:red"), ("negative", "tab:blue")):
            values = graphs[key][row]
            profile = [np.nanmean(np.diagonal(values, -lag)) for lag in lags]
            ax.plot(lags * segment_ms, profile, color=colour, label=f"{key} edges")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="even share")
        ax.set_xlabel("lag t - s (ms)")
        ax.set_ylabel("relative importance")
        ax.set_title(f"{title}\nimportance by lag", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Learned segment graph (test windows)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, path)
