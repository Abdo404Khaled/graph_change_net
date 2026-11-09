"""Tables and figures of the paper's results section, written only from result files.

Writes into the output folder:

- ``tables/table_rotation.tex``: GraphChangeNet and the 13 baseline configurations of the rotation
  (28 nights; pooled macro-F1 and κ with Holm marks from ``rotation_stats.json``, N1 F1 and the mean
  over the seven groups from each run's ``metrics.json``), plus the five-fold value where it exists;
- ``tables/table_ablation.tex``: (A) the runs that built the final model and (B) the removals from it,
  five folds each (mean ± SD and folds below GraphChangeNet); (C) the mechanism ablations of the
  single split with staging (Holm marks from ``stats.json``) and stage-1 measures on the test nights
  (``stage1_stats.json``, ``stage1_stats_t1.json``), and the nearest visible electrode;
- ``figures/fig_missing.pdf``: variance explained of a hidden electrode against the number of hidden
  neighbours on the test nights;
- ``figures/fig_buildup.pdf``: five-fold macro-F1 of the runs in (A) and (B);
- ``figures/fig_views.pdf``: stable graphs of GraphChangeNet and of the model trained without the change,
  each in 3-D and top / side / front views, positive and negative edges in separate rows;
- ``figures/fig_cost.pdf``: (a) unified macro-F1 against operations per epoch on the same input, marker
  area by parameters, Pareto front; (b) operations and peak GPU memory per epoch against the number of
  electrodes (synthetic input), from ``cost_baselines.json`` and ``cost_static.json``;
- ``figures/fig_inputs.pdf``: κ of every model tested on other preparations of the same nights, drawn
  with vector cells.

argv: output folder (``tables/`` and ``figures/`` are written into it).
"""
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NOTES = REPO / "analysis/results"
RUNS, ABL, BASE = REPO / "runs", REPO / "ablations/results", REPO / "baselines/results"
PAPER = Path(sys.argv[1])
TABLES, FIGURES = PAPER / "tables", PAPER / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, EDGE, GRID, OURS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9", "#d62f2f"
COLUMN = 3.45
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.edgecolor": EDGE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.labelsize": 7, "mathtext.fontset": "dejavusans"})
ROTATION_ROWS = [
    ("AnySleep", "N$^\\dagger$", "AnySleep (native, zero-shot)", "anysleep", "native"),
    ("AnySleep", "C$^\\dagger$", "AnySleep (causal, zero-shot)", "anysleep", "causal"),
    ("U-Sleep", "N$^\\dagger$", "U-Sleep (native, zero-shot)", "usleep", "native"),
    ("Hypno-PC", "N", "Hypno-PC (native)", "hypnopc", "native"),
    ("GraphSleepNet", "N", "GraphSleepNet (native)", "graphsleepnet", "native"),
    ("GraphSleepNet", "C", "GraphSleepNet (causal)", "graphsleepnet", "causal"),
    ("GraphSleepNet", "CNN--2S$^\\ddagger$", "GraphSleepNet (CNN-2S)", "graphsleepnet", "cnn2s"),
    ("GraphSleepNet-d.", "N", "GraphSleepNet-dist. (native)", "graphsleepnet_distance", "native"),
    ("MSTGCN", "N$^\\ddagger$", "MSTGCN (native)", "mstgcn", "native"),
    ("MSTGCN", "C$^\\ddagger$", "MSTGCN (causal)", "mstgcn", "causal"),
    ("ProductGSN", "N", "ProductGraphSleepNet (native)", "productgraphsleepnet", "native"),
    ("ProductGSN", "C", "ProductGraphSleepNet (causal)", "productgraphsleepnet", "causal"),
    ("ProductGSN", "CNN--2S$^\\ddagger$", "ProductGraphSleepNet (CNN-2S)", "productgraphsleepnet", "cnn2s"),
]
CNN_FRONTEND = 69_952


def read(path: Path) -> dict:
    """Parsed JSON file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict
        Content.
    """
    return json.loads(Path(path).read_text())


def number(value: float, digits: int = 3) -> str:
    """A value with a LaTeX minus sign.

    Parameters
    ----------
    value : float
        Value to format.
    digits : int
        Decimals.

    Returns
    -------
    str
        Formatted value.
    """
    return f"{value:.{digits}f}".replace("-", "$-$")


def millions(count: int) -> str:
    """Parameter count in millions, two decimals below 10 M and one above.

    Parameters
    ----------
    count : int
        Number of parameters.

    Returns
    -------
    str
        Formatted count.
    """
    value = count / 1e6
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


SHADES = ("rankA", "rankB", "rankC")


def shade(lines: list, columns: dict, separator: str = "\\multicolumn") -> list:
    """Shade the three best values of each column, darkest for the best, as in the paper's tables.

    Rows after the first ``\\midrule`` are compared; a line that starts with ``separator``
    starts a new group, so groups are ranked separately. A cell's value is
    its first number (a mean in ``mean ± SD`` cells); ``---`` cells are skipped. Equal values share
    a shade.

    Parameters
    ----------
    lines : list of str
        Table lines, one row per line.
    columns : dict
        Column index to ``"max"`` (higher is better) or ``"min"``.
    separator : str
        Start of the lines that separate groups.

    Returns
    -------
    list of str
        The lines with ``\\cellcolor`` added to the shaded cells.
    """
    def value(cell: str) -> float | None:
        found = re.search(r"(\$-\$|-)?(\d+\.\d+)", cell)
        if found is None:
            return None
        return -float(found.group(2)) if found.group(1) else float(found.group(2))

    start = next(i for i, line in enumerate(lines) if line.startswith("\\midrule")) + 1
    groups, current = [], []
    for index in range(start, len(lines)):
        line = lines[index]
        if line.lstrip().startswith(separator) or line.startswith("\\bottomrule"):
            groups.append(current)
            current = []
        elif "&" in line:
            current.append(index)
    groups.append(current)
    out = list(lines)
    cells = {i: lines[i].rstrip().removesuffix("\\\\").split(" & ") for i in range(len(lines)) if "&" in lines[i]}
    for group in groups:
        for column, direction in columns.items():
            values = {i: value(cells[i][column]) for i in group if column < len(cells[i])}
            values = {i: v for i, v in values.items() if v is not None}
            ranked = sorted(set(values.values()), reverse=direction == "max")[:3]
            for i, v in values.items():
                if v in ranked:
                    cells[i][column] = f"\\cellcolor{{{SHADES[ranked.index(v)]}}}{cells[i][column]}"
    for i, parts in cells.items():
        if i >= start:
            out[i] = " & ".join(parts) + " \\\\"
    return out


def rotation_table() -> None:
    """Rotation over 28 nights, every group: macro-F1 per group, mean ± SD, pooled MF1 and κ with Holm marks,
    N1 F1, κ on the two artifact-inflated nights (EPCTL18, EPCTL25) and the five-fold macro-F1."""
    stats = read(NOTES / "rotation_stats.json")
    groups = range(7)
    first = stats["AnySleep (native, zero-shot)"]

    def row(name, conf, count, tests, pooled_f1, pooled_k, mark_f, mark_k, art, five):
        f1 = [t["macro_f1"] for t in tests]
        cells = [number(v) for v in f1]
        return (f"{name} & {conf} & {count} & " + " & ".join(cells)
                + f" & {number(statistics.mean(f1))}$\\pm${statistics.pstdev(f1):.3f} & {number(pooled_f1)}{mark_f} & "
                f"{number(pooled_k)}{mark_k} & {number(statistics.mean(t['f1_N1'] for t in tests))} & "
                f"{number(art[0], 2)}, {number(art[1], 2)} & {five} \\\\")

    lines = ["\\begin{tabular}{@{}llr" + "c" * 7 + "cccccc@{}}", "\\toprule",
             " & & & \\multicolumn{7}{c}{MF1 per rotation group (4 test nights each)} & & \\multicolumn{2}{c}{Pooled, 28 nights} & & & \\\\",
             "\\cmidrule(lr){4-10}\\cmidrule(lr){12-13}",
             "Model & Conf. & Params & rot0 & rot1 & rot2 & rot3 & rot4 & rot5 & rot6 & Mean$\\pm$SD & MF1 & $\\kappa$ & N1 F1 & $\\kappa$ EPCTL18, 25 & 5-fold MF1 \\\\",
             "\\midrule"]
    ours = [read(RUNS / f"exp15_rotation_rot{g}/metrics.json")["test"] for g in groups]
    folds = [read(RUNS / f"exp15_final_fold{k}/metrics.json")["test"] for k in range(5)]
    art = [first["per_night"][n]["exp15"] for n in ("EPCTL18", "EPCTL25")]
    lines.append(row("\\textbf{GraphChangeNet}", "U, causal", "2.03", ours, first["exp15_macro_f1"], first["exp15_kappa"],
                     "", "", art, number(statistics.mean(t["macro_f1"] for t in folds))))
    lines.append("\\midrule")
    for name, conf, key, model, config in sorted(ROTATION_ROWS, key=lambda r: -stats[r[2]]["baseline_macro_f1"]):
        entry = stats[key]
        runs = [read(BASE / model / f"{config}_rot{g}/metrics.json") for g in groups]
        count = millions(runs[0]["parameters"] + (CNN_FRONTEND if config == "cnn2s" else 0))
        art = [entry["per_night"][n]["baseline"] for n in ("EPCTL18", "EPCTL25")]
        five = number(read(BASE / model / config / "metrics.json")["test"]["macro_f1"]) if model in ("anysleep", "usleep") else "---"
        lines.append(row(name, conf, count, [r["test"] for r in runs], entry["baseline_macro_f1"], entry["baseline_kappa"],
                         "$^{*}$" if entry["holm_p_macro_f1"] < 0.05 else "", "$^{*}$" if entry["holm_p_kappa"] < 0.05 else "",
                         art, five))
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines = shade(lines, {c: "max" for c in (*range(3, 14), 15)})
    (TABLES / "table_rotation.tex").write_text("\n".join(lines) + "\n")


def single_split_table() -> None:
    """Table II: GraphChangeNet and every published model in its native and the unified configuration.

    Single split, all 4,462 scored test epochs; GraphChangeNet first, then the published models sorted
    by their 28-night macro-F1 and, for configurations not run in the rotation, by single-split
    macro-F1; the last column is the pooled macro-F1 over the 28 rotation nights of every
    configuration that ran in the rotation; the three best values of each column shaded; Holm marks for macro-F1 and κ
    from ``stats.json`` (paired bootstrap over the test nights).
    """
    names = {"anysleep": "AnySleep", "usleep": "U-Sleep", "deepsleepnet": "DeepSleepNet", "cbramod": "CBraMod",
             "hypnopc": "Hypno-PC", "graphsleepnet": "GraphSleepNet", "graphsleepnet_distance": "GraphSleepNet-d.",
             "mstgcn": "MSTGCN", "productgraphsleepnet": "ProductGSN"}
    graph = {"graphsleepnet", "graphsleepnet_distance", "mstgcn", "productgraphsleepnet"}
    stats = read(NOTES / "stats.json")["baselines"]
    rotation = read(NOTES / "rotation_stats.json")
    pooled = {f"{model} ({config})": number(rotation[key]["baseline_macro_f1"])
              for _, _, key, model, config in ROTATION_ROWS}
    reference = read(RUNS / "exp15_final_fold0/metrics.json")
    rows = [("\\textbf{GraphChangeNet}", "U", True, reference["parameters"], reference["test"], None)]
    for key, label in names.items():
        for setting in ("native", "ours"):
            m = read(BASE / key / setting / "metrics.json")
            tag = ("N$^\\dagger$" if key in ("anysleep", "usleep") else "N") if setting == "native" else "U"
            rows.append((label, tag, key in graph, m["parameters"], m["test"], f"{key} ({setting})"))
    for key, setting, tag in (("anysleep", "causal", "C$^\\dagger$"), ("graphsleepnet", "causal", "C"),
                              ("mstgcn", "causal", "C"), ("productgraphsleepnet", "causal", "C"),
                              ("graphsleepnet", "cnn2s", "CNN--2S"), ("productgraphsleepnet", "cnn2s", "CNN--2S")):
        m = read(BASE / key / setting / "metrics.json")
        count = m["parameters"] + (CNN_FRONTEND if setting == "cnn2s" else 0)
        rows.append((names[key], tag, key in graph, count, m["test"], f"{key} ({setting})"))
    rows[1:] = sorted(rows[1:], key=lambda r: (-float(pooled.get(r[5], "0")), -r[4]["macro_f1"]))
    lines = ["\\begin{tabular}{@{}lccrcccc@{}}", "\\toprule",
             " & & & & \\multicolumn{3}{c}{Single split} & 28 nights \\\\",
             "\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}",
             "Model & Conf. & Graph & Params (M) & MF1 & $\\kappa$ & N1 F1 & MF1 \\\\", "\\midrule"]
    for label, tag, is_graph, count, test, key in rows:
        s_f = "$^{*}$" if key and stats.get(key, {}).get("p_macro_f1_holm", 1) < 0.05 else ""
        s_k = "$^{*}$" if key and stats.get(key, {}).get("p_kappa_holm", 1) < 0.05 else ""
        whole = number(rotation["AnySleep (native, zero-shot)"]["exp15_macro_f1"]) if key is None else pooled.get(key, "---")
        cells = [number(test["macro_f1"]) + s_f, number(test["kappa"]) + s_k, number(test["f1_N1"]), whole]
        mark = "\\checkmark" if is_graph else ""
        lines.append(f"{label} & {tag} & {mark} & {millions(count)} & " + " & ".join(cells) + " \\\\")
        if key is None:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines = shade(lines, {4: "max", 5: "max", 6: "max", 7: "max"})
    (TABLES / "table_baselines.tex").write_text("\n".join(lines) + "\n")


def fold_scores(prefix: Path, name: str) -> tuple[list, list, list, int]:
    """Macro-F1, κ and N1 F1 of the five folds of one run, and its parameter count.

    Parameters
    ----------
    prefix : Path
        Folder that holds the fold folders.
    name : str
        Run name without the fold suffix.

    Returns
    -------
    tuple
        Macro-F1 list, κ list, N1 list, parameters.
    """
    metrics = [read(prefix / f"{name}_fold{k}/metrics.json") for k in range(5)]
    return ([m["test"]["macro_f1"] for m in metrics], [m["test"]["kappa"] for m in metrics],
            [m["test"]["f1_N1"] for m in metrics], metrics[0]["parameters"])


def ablation_table() -> list:
    """Table III: five-fold build-up and removals, single-split mechanisms with stage-1 measures.

    Returns
    -------
    list
        ``(label, macro-F1 per fold)`` of the five-fold rows, for the build-up figure.
    """
    reference_f1 = fold_scores(RUNS, "exp15_final")[0]
    built = [("A1 Level 1 (electrode graph), GRU over the seconds", "final5_signed_gcn"),
             ("A2 + Level 2 (Transformer, segment graph)", "final11_lag_graph_transformer"),
             ("A3 + pooling, stage-1 tasks, lights-on epochs", "exp14a_scored_epochs"),
             ("A4 + Level 3 (epoch context)", "exp14b_epoch_context"),
             ("A5 + stage 1 that trains: \\textbf{GraphChangeNet}", "exp15_final")]
    removed = [("B1 without Level 3", "exp15b_no_context"),
               ("B2 without spectral and next-second tasks", "exp15c_reconstruction_only"),
               ("B3 without stage 1", "exp15a_no_stage1")]
    stage1 = read(NOTES / "stage1_stats.json")
    t1 = read(NOTES / "stage1_stats_t1.json")
    holm = read(NOTES / "stats.json")["components"]
    cols = 10
    lines = [f"\\begin{{tabular}}{{@{{}}lrccccccc c@{{}}}}".replace(" c@", "c@"), "\\toprule",
             " & & \\multicolumn{3}{c}{Staging (test)} & \\multicolumn{2}{c}{Stage-1 loss / trivial $\\downarrow$} "
             "& \\multicolumn{3}{c}{VE, $k$ neighbours also hidden $\\uparrow$} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}",
             "Run & Params (M) & MF1 & $\\kappa$ & N1 F1 & recon. & spectrum & $k=0$ & $k=5$ & $k=10$ \\\\",
             "\\midrule",
             f"\\multicolumn{{{cols}}}{{@{{}}l}}{{\\emph{{(a) Building the model, five folds: mean $\\pm$ SD (folds below GraphChangeNet)}}}} \\\\"]
    figure_rows = []
    for label, name in built + removed:
        f1, kappa, n1, count = fold_scores(RUNS, name)
        below = sum(a < b for a, b in zip(f1, reference_f1))
        tail = "" if name == "exp15_final" else f" ({below}/5)"
        if name == removed[0][1]:
            lines.append(f"\\multicolumn{{{cols}}}{{@{{}}l}}{{\\emph{{(b) Removing one part of GraphChangeNet, five folds}}}} \\\\")
        lines.append(f"\\quad {label} & {millions(count)} & {number(statistics.mean(f1))}$\\pm${statistics.pstdev(f1):.3f}{tail} "
                     f"& {number(statistics.mean(kappa))} & {number(statistics.mean(n1))} & --- & --- & --- & --- & --- \\\\")
        figure_rows.append((label.split(" ")[0], f1))
    lines.append(f"\\midrule\n\\multicolumn{{{cols}}}{{@{{}}l}}{{\\emph{{(c) Mechanisms of Level 1, single split; stage 1 on the 4,462 test epochs}}}} \\\\")
    reference = stage1["Exp 15"]
    exp15 = read(RUNS / "exp15_final_fold0/metrics.json")["test"]

    def stage1_cells(row: dict | None) -> str:
        if row is None:
            values = [reference["reconstruction"], reference["spectral"], reference["ve_k0"],
                      reference["ve_k5"], reference["ve_k10"]]
            return " & ".join(number(v) for v in values)
        out = []
        for metric in ("reconstruction", "spectral", "ve_k0", "ve_k5", "ve_k10"):
            cell = row[metric]
            big = abs(cell["delta"]) >= (0.05 * reference[metric] if not metric.startswith("ve") else 0.02)
            mark = "$^{\\circ}$" if cell["p_holm"] < 0.05 and big else ""
            out.append(number(cell["value"]) + mark)
        return " & ".join(out)

    lines.append(f"\\quad \\textbf{{GraphChangeNet}} & 2.03 & {number(exp15['macro_f1'])} & "
                 f"{number(exp15['kappa'])} & {number(exp15['f1_N1'])} & {stage1_cells(None)} \\\\")
    mechanisms = [("no electrode graph", "no_electrode_graph_fold0", "no_electrode_graph"),
                  ("one graph per 30\\,s epoch", "epoch_graph_fold0", "epoch_graph"),
                  ("stable graph only ($\\Delta_t=0$)", "stable_only_fold0", "stable_only"),
                  ("change without memory ($\\alpha=1$)", "no_memory_fold0", "no_memory"),
                  ("fixed scalp-distance graph", "distance_graph_fold0", "distance_graph")]
    for label, folder, key in mechanisms:
        m = read(ABL / folder / "metrics.json")
        mark = "$^{*}$" if holm[key]["p_macro_f1_holm"] < 0.05 else ""
        lines.append(f"\\quad {label} & {millions(m['parameters'])} & {number(m['test']['macro_f1'])}{mark} & "
                     f"{number(m['test']['kappa'])} & {number(m['test']['f1_N1'])} & {stage1_cells(stage1['variants'][key])} \\\\")
    unsigned = ABL / "unsigned_graph_fold0"
    if (unsigned / "metrics.json").exists():
        m = read(unsigned / "metrics.json")
        size, test = millions(m["parameters"]), (m["test"]["kappa"], m["test"]["macro_f1"], m["test"]["f1_N1"])
    else:
        log = (REPO / "logs/ablation_unsigned_graph_fold0.log").read_text(errors="replace")
        size = "2.03"
        test = [float(v) for v in re.findall(r"TEST: \{'kappa': ([0-9.]+), 'macro_f1': ([0-9.]+).*?'f1_N1': ([0-9.]+)", log)[-1]]
    lines.append(f"\\quad trained without edge signs & {size} & {number(test[1])} & {number(test[0])} & "
                 f"{number(test[2])} & {stage1_cells(t1['variants']['unsigned_graph'])} \\\\")
    near = t1["nearest_visible"]
    lines.append("\\quad nearest visible electrode (no model) & --- & --- & --- & --- & --- & --- & "
                 + " & ".join(number(near[f"ve_k{k}"]["value"]) for k in (0, 5, 10)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines = shade(lines, {2: "max", 3: "max", 4: "max", 5: "min", 6: "min", 7: "max", 8: "max", 9: "max"},
                  separator="\\midrule")
    (TABLES / "table_ablation.tex").write_text("\n".join(lines) + "\n")
    return figure_rows


def missing_figure() -> None:
    """Variance explained against the number of hidden neighbours, test nights, same masks."""
    stage1 = read(NOTES / "stage1_stats.json")
    t1 = read(NOTES / "stage1_stats_t1.json")
    ks = [0, 5, 10]
    series = [("GraphChangeNet", [stage1["Exp 15"][f"ve_k{k}"] for k in ks], OURS, "-", "o"),
              ("fixed distance graph", [stage1["variants"]["distance_graph"][f"ve_k{k}"]["value"] for k in ks], "#2a78d6", "-", "s"),
              ("one graph per epoch", [stage1["variants"]["epoch_graph"][f"ve_k{k}"]["value"] for k in ks], "#eda100", "-", "^"),
              ("stable graph only", [stage1["variants"]["stable_only"][f"ve_k{k}"]["value"] for k in ks], "#6250d6", "--", "v"),
              ("trained without signs", [t1["variants"]["unsigned_graph"][f"ve_k{k}"]["value"] for k in ks], "#1baf7a", "--", "D"),
              ("nearest visible electrode", [t1["nearest_visible"][f"ve_k{k}"]["value"] for k in ks], MUTED, ":", "x")]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.95))
    for label, values, colour, style, marker in series:
        ax.plot(ks, values, style, color=colour, marker=marker, ms=3.2, lw=1.1, label=label)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(ks)
    ax.set_xlabel("nearest neighbours hidden with the electrode ($k$)")
    ax.set_ylabel("variance explained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
              columnspacing=0.8, fontsize=5.8)
    fig.savefig(FIGURES / "fig_missing.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def buildup_figure(rows: list) -> None:
    """Five-fold macro-F1 of the build-up and removal runs, fold dots and mean.

    Parameters
    ----------
    rows : list
        ``(label, macro-F1 per fold)`` from :func:`ablation_table`.
    """
    fig, ax = plt.subplots(figsize=(COLUMN, 1.55))
    x = np.arange(len(rows))
    for i, (label, values) in enumerate(rows):
        colour = OURS if label == "A5" else (MUTED if label.startswith("B") else INK2)
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), values, s=6, color=colour, alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.mean(values)] * 2, color=colour, lw=1.6)
    ax.axhline(np.mean(rows[4][1]), color=OURS, lw=0.6, ls=":", zorder=0)
    ax.axvline(4.5, color=GRID, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A1\nL1", "A2\n+L2", "A3\n+pool", "A4\n+L3", "A5\n+S1", "B1\n$-$L3",
                        "B2\n$-$tasks", "B3\n$-$S1"], fontsize=6)
    ax.set_ylabel("macro-F1 (5 folds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "fig_buildup.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def inputs_figure() -> None:
    """Every model trained on 0.5-45 Hz input and tested on other preparations, as a vector heatmap."""
    from matplotlib.colors import LinearSegmentedColormap
    sequential = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#1c5cab", "#0d366b"])
    iv = read(NOTES / "variants/input_variants.json")
    bv1, bv2 = (read(NOTES / "baseline_variants/baseline_variants.json"),
                read(NOTES / "baseline_variants/baseline_variants_2.json"))
    columns = [("as\ntrained", None), ("20 Hz\nlow-p.", "0.5-20 Hz"), ("30 Hz\nlow-p.", "0.5-30 Hz"),
               ("no\nlow-p.", "0.5 Hz high-pass + notches, no low-pass"),
               ("no\nnotch", "0.5 Hz high-pass only (no notch, no low-pass)"),
               ("no\nhigh-p.", "notches 60-420 Hz only (no high-pass, no low-pass)"), ("raw", "raw (no band-pass, no notch)")]
    rows = [("GraphChangeNet", "exp15"), ("AnySleep", "anysleep"), ("U-Sleep", "usleep"), ("CBraMod", "cbramod"),
            ("DeepSleepNet", "deepsleepnet"), ("MSTGCN", "mstgcn"), ("GraphSleepNet", "graphsleepnet"),
            ("GraphSleepNet-d.", "graphsleepnet_distance"), ("ProductGSN", "productgraphsleepnet")]
    grid = np.full((len(rows), len(columns)), np.nan)
    for i, (_, key) in enumerate(rows):
        for j, (_, cond) in enumerate(columns):
            if cond is None:
                path = RUNS / "exp15_final_fold0" if key == "exp15" else BASE / key / "ours"
                grid[i, j] = read(path / "metrics.json")["test"]["kappa"]
            elif key == "exp15":
                grid[i, j] = (iv.get(cond) or (bv2.get("exp15") or {}).get(cond))["kappa"]
            else:
                for source in (bv1, bv2):
                    if cond in (source.get(key) or {}):
                        grid[i, j] = source[key][cond]["kappa"]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    ax.pcolormesh(np.arange(len(columns) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5, grid, cmap=sequential,
                  vmin=0, vmax=0.85, edgecolors="none", rasterized=False)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if grid[i, j] > 0.5 else INK, weight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(columns)), [c[0] for c in columns], fontsize=6.3)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.6)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.tick_params(length=0, pad=2)
    ax.axvline(0.5, color="white", lw=2.5)
    for y in (0.5, 4.5):
        ax.axhline(y, color="white", lw=2.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(FIGURES / "fig_inputs.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def head_outline(ax, view: str, xs: np.ndarray, ys: np.ndarray) -> None:
    """Head silhouette behind a 2-D electrode projection (as in ``results_figures.py``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Panel.
    view : str
        ``top``, ``side`` or ``front``.
    xs, ys : np.ndarray
        Projected electrode positions.
    """
    from matplotlib.patches import Ellipse
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    w, h = (xs.max() - xs.min()) * 1.16, (ys.max() - ys.min()) * 1.16
    style = dict(fc="#f7f6f2", ec=MUTED, lw=0.6, zorder=0)
    ax.add_patch(Ellipse((cx, cy), w, h, **style))
    if view == "top":
        ax.add_patch(plt.Polygon([[cx - 0.08 * w, cy + 0.49 * h], [cx, cy + 0.6 * h], [cx + 0.08 * w, cy + 0.49 * h]], **style))
        for sign in (-1, 1):
            ax.add_patch(Ellipse((cx + sign * 0.52 * w, cy), 0.08 * w, 0.2 * h, **style))
    elif view == "side":
        ax.add_patch(plt.Polygon([[cx - 0.49 * w, cy + 0.05 * h], [cx - 0.6 * w, cy - 0.06 * h],
                                  [cx - 0.47 * w, cy - 0.12 * h]], **style))
    else:
        for sign in (-1, 1):
            ax.add_patch(Ellipse((cx + sign * 0.52 * w, cy - 0.05 * h), 0.08 * w, 0.22 * h, **style))
    ax.set_xlim(cx - 0.68 * w, cx + 0.68 * w)
    ax.set_ylim(cy - 0.62 * h, cy + 0.68 * h)
    ax.set_aspect("equal")
    ax.axis("off")


def views_figure() -> None:
    """Stable graphs in 3-D and in top / side / front views, positive and negative edges in separate rows.

    Each electrode's three strongest edges of each sign are drawn, each pair once; width and opacity
    follow the weight relative to the strongest drawn edge of that sign. The text under the 3-D panel is
    the mean length of the drawn edges in units of the mean inter-electrode distance.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    from data import find_montage, read_montage
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _, coords = read_montage(find_montage(REPO / "data/anphy/ANPHY-Sleep/osfstorage"))
    coords = coords - coords.mean(0)
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    mean_distance = distance[np.triu_indices(len(coords), 1)].mean()
    graphs = [("GraphChangeNet: $L^S$", np.load(RUNS / "exp15_final_fold0/graphs.npz")["stable"]),
              ("trained without $\\Delta_t$: $L^S$", np.load(ABL / "stable_only_fold0/graphs.npz")["stable"])]
    projections = [("top", lambda q: (-q[:, 1], q[:, 0])), ("side", lambda q: (-q[:, 0], q[:, 2])),
                   ("front", lambda q: (q[:, 1], q[:, 2]))]
    fig = plt.figure(figsize=(7.16, 1.95))
    grid = fig.add_gridspec(2, 9, width_ratios=[0.46] + [1.2, 1, 1, 1] * 2, left=0.0, right=1.0, top=0.88,
                            bottom=0.08, wspace=0.03, hspace=0.12)
    for row, (sign, colour, name) in enumerate(((1, "#d62f2f", "positive\nedges"), (-1, "#2a78d6", "negative\nedges"))):
        label = fig.add_subplot(grid[row, 0])
        label.axis("off")
        label.text(0.5, 0.5, name, ha="center", va="center", fontsize=7, color=colour, weight="bold")
        for g, (title, matrix) in enumerate(graphs):
            signed = 0.5 * (matrix + matrix.T)
            strength = np.where(np.eye(len(signed), dtype=bool), -np.inf, sign * signed)
            pairs = sorted({(min(i, int(j)), max(i, int(j))) for i, r in enumerate(np.argsort(-strength, 1)[:, :3])
                            for j in r if strength[i, j] > 0})
            rows_, cols_ = map(np.asarray, zip(*pairs))
            weights = np.abs(signed[rows_, cols_])
            top = weights.max()
            order = np.argsort(weights)
            ax3 = fig.add_subplot(grid[row, 1 + 4 * g], projection="3d")
            for e in order:
                r = weights[e] / top
                ax3.plot(*[[coords[rows_[e], d], coords[cols_[e], d]] for d in range(3)], color=colour,
                         lw=0.15 + 0.75 * r, alpha=0.12 + 0.68 * r ** 2)
            ax3.scatter(*coords.T, s=1.4, color=INK, depthshade=False, lw=0)
            ax3.view_init(elev=22, azim=-60)
            ax3.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])), zoom=1.45)
            ax3.set_axis_off()
            ax3.text2D(0.5, 0.04, f"mean length {distance[rows_, cols_].mean() / mean_distance:.2f}",
                       transform=ax3.transAxes, ha="center", va="top", fontsize=6, color=colour,
                       bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))

            for v, (vname, view) in enumerate(projections):
                ax = fig.add_subplot(grid[row, 2 + 4 * g + v])
                xs, ys = view(coords)
                head_outline(ax, vname, xs, ys)
                for e in order:
                    r = weights[e] / top
                    ax.plot([xs[rows_[e]], xs[cols_[e]]], [ys[rows_[e]], ys[cols_[e]]], color=colour,
                            lw=0.15 + 0.75 * r, alpha=0.12 + 0.68 * r ** 2, zorder=2)
                ax.scatter(xs, ys, s=1.6, color=INK, lw=0, zorder=3)

    for g, (title, _) in enumerate(graphs):
        for v, vname in enumerate(("3-D", "top", "side", "front")):
            box = grid[0, 1 + 4 * g + v].get_position(fig)
            fig.text((box.x0 + box.x1) / 2, 0.905, vname, ha="center", va="center", fontsize=6.5, color=INK2)
        x0 = grid[0, 1 + 4 * g].get_position(fig).x0
        x1 = grid[0, 4 + 4 * g].get_position(fig).x1
        fig.text((x0 + x1) / 2, 0.975, title, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        if g:
            fig.add_artist(plt.Line2D([x0 - 0.004] * 2, [0.02, 0.99], transform=fig.transFigure, color=GRID, lw=0.8))
    fig.savefig(FIGURES / "fig_views.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def cost_figure() -> None:
    """Accuracy against compute on the same input, and compute and memory against electrode count."""
    cost = read(NOTES / "cost_baselines.json")
    sweep = read(NOTES / "cost_static.json")["electrode_sweep"]
    names = {"graphchangenet": "GraphChangeNet", "anysleep": "AnySleep", "usleep": "U-Sleep", "cbramod": "CBraMod",
             "deepsleepnet": "DeepSleepNet", "graphsleepnet": "GraphSleepNet", "graphsleepnet_distance": "GSN-d.",
             "mstgcn": "MSTGCN", "productgraphsleepnet": "ProductGSN"}
    points = []
    for key, label in names.items():
        path = RUNS / "exp15_final_fold0" if key == "graphchangenet" else BASE / key / "ours"
        points.append((label, cost[key]["gflops_per_epoch_amortised"], read(path / "metrics.json")["test"]["macro_f1"],
                       cost[key]["parameters"]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLUMN, 1.42), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.55})
    front = sorted(points, key=lambda q: q[1])
    best, line = -1.0, []
    for label, g, f, n in front:
        if f > best:
            line.append((g, f)); best = f
    ax.plot(*zip(*line), color=GRID, lw=1.2, zorder=1, ls="--")
    offsets = {"GraphChangeNet": (-34, 5), "AnySleep": (5, -9), "CBraMod": (-12, 5), "U-Sleep": (4, -6),
               "DeepSleepNet": (5, -2), "GraphSleepNet": (4, -6), "GSN-d.": (4, 1), "MSTGCN": (4, -6), "ProductGSN": (5, -2)}
    for label, g, f, n in points:
        ours = label == "GraphChangeNet"
        ax.scatter(g, f, s=6 + 9 * np.log10(n / 1e4), color=OURS if ours else INK2, alpha=0.9 if ours else 0.55,
                   lw=0, zorder=3)
        ax.annotate(f"{label} {f:.3f}", (g, f), xytext=offsets[label], textcoords="offset points", fontsize=5.6,
                    color=OURS if ours else INK2, weight="bold" if ours else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 900)
    ax.set_ylim(0.73, 0.865)
    ax.set_xlabel("GFLOPs per epoch")
    ax.set_ylabel("macro-F1 (our input)")
    ax.set_title("(a)", fontsize=7, loc="left", pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    c = [q["electrodes"] for q in sweep]
    bx.plot(c, [q["gflops_per_epoch"] for q in sweep], "-o", color=OURS, ms=2.5, lw=1.1)
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xticks([19, 64, 256]); bx.set_xticklabels(["19", "64", "256"], fontsize=5.6)
    bx.axvline(83, color=GRID, lw=0.8, ls=":", zorder=0)
    bx.text(83, 0.93, "83", fontsize=5.6, color=MUTED, ha="center", transform=bx.get_xaxis_transform())
    bx.minorticks_off()
    bx.set_xlabel("electrodes")
    bx.set_ylabel("GFLOPs", color=OURS, labelpad=1)
    mem = bx.twinx()
    mem.plot(c, [q["peak_mib_streaming"] for q in sweep], "--s", color=INK2, ms=2.2, lw=1.0)
    mem.set_ylabel("memory (MiB)", color=INK2, labelpad=1)
    mem.set_ylim(350, 650)
    flops, peak = [q["gflops_per_epoch"] for q in sweep], [q["peak_mib_streaming"] for q in sweep]
    bx.annotate(f"{flops[0]:.2f}", (c[0], flops[0]), xytext=(3, -2), textcoords="offset points", fontsize=5.6,
                color=OURS, weight="bold")
    bx.annotate(f"{flops[-1]:.1f}", (c[-1], flops[-1]), xytext=(-14, 2), textcoords="offset points", fontsize=5.6,
                color=OURS, weight="bold")
    mem.annotate(f"{peak[0]:.0f}", (c[0], peak[0]), xytext=(0, 4), textcoords="offset points", fontsize=5.6,
                 color=INK2)
    mem.annotate(f"{peak[-1]:.0f}", (c[-1], peak[-1]), xytext=(-20, -9), textcoords="offset points", fontsize=5.6,
                 color=INK2)
    bx.set_title("(b)", fontsize=7, loc="left", pad=2)
    for a in (bx, mem):
        a.spines["top"].set_visible(False)
    fig.savefig(FIGURES / "fig_cost.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    rotation_table()
    single_split_table()
    buildup_figure(ablation_table())
    missing_figure()
    inputs_figure()
    views_figure()
    cost_figure()
    print("written:", sorted(p.name for p in TABLES.glob("table_*.tex")), "fig_missing.pdf fig_buildup.pdf")
