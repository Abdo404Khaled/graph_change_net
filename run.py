from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from config import PLAN, QUICK, Config, dotted, git_state, read_plan, read_yaml, resolve
from data import build_loaders, load_dataset, split_subjects
from evaluate import (
    collect_graphs,
    context_report,
    collect_segment_graphs,
    evaluate_shuffled,
    format_reconstruction,
    predict,
    reconstruction_report,
    rolling_report,
    scores,
    segment_graph_report,
)
from model import GraphChangeNet, load_pretrained_parts, load_weights
from train import baselines, classify, pretrain

warnings.filterwarnings("ignore")

DESCRIPTION = """Run experiments end to end: data, stage 1, stage 2, evaluation, figures.

Configuration layers, lowest first:
  configs/default.yaml          every setting
  configs/<experiment>.yaml     what one experiment changes (may name a 'base' file)
  experiments.yaml              the plan: which experiments to run, 'common' and per-entry 'set'
  --quick, --set key=value      command-line overrides

examples:
  python run.py                                    run every enabled entry of experiments.yaml
  python run.py --only exp6_segment_position       run selected plan entries
  python run.py --config configs/exp5_signed_gcn.yaml --set finetune.lr=5e-4 --name exp5_lr
  python run.py --only exp5_signed_gcn --quick     few-minute pipeline check
  python run.py --only exp5_signed_gcn --print-config
  python run.py --config exp5_signed_gcn --pretrained P.pt --checkpoint C.pt   evaluate weights
  python run.py --only exp8_segment_graph --set split.n_folds=5 split.fold=0   one k-fold run
  python run.py --fold-summary exp7_signed_attention exp8_segment_graph       mean +- std over folds
  python run.py --only exp14a_scored_epochs --prepare-data                  build the data cache only
"""


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=PLAN, help="plan file (default: %(default)s)")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run these plan entries, enabled or not")
    parser.add_argument("--config", help="run one experiment file instead of the plan")
    parser.add_argument("--name", help="run name, with --config (default: the file name)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override settings, e.g. finetune.lr=5e-4 model.position.name=segment")
    parser.add_argument("--quick", action="store_true",
                        help=f"apply {QUICK} (1 + 2 epochs) and add '_quick' to the run names")
    parser.add_argument("--pretrained", type=Path, help="stage 1 weights; skips pretraining")
    parser.add_argument("--checkpoint", type=Path, help="stage 2 weights; skips all training")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    parser.add_argument("--list", action="store_true", help="list the plan and exit")
    parser.add_argument("--fold-summary", nargs="+", metavar="NAME",
                        help="summarise the finished k-fold runs NAME_fold<k> and exit")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved configs and exit")
    parser.add_argument("--prepare-data", action="store_true",
                        help="build the preprocessing cache of the selected runs and exit")
    return parser.parse_args()


def command_line_layers(args: argparse.Namespace) -> list[dict]:
    """The override layers that come from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options.

    Returns
    -------
    list of dict
        ``configs/quick.yaml`` when ``--quick`` is given, then the ``--set`` values.
    """
    layers = [read_yaml(QUICK)] if args.quick else []
    if args.set:
        layers.append(dotted(args.set))
    if args.no_plots:
        layers.append({"evaluation": {"plots": False}})
    return layers


def selected_configs(args: argparse.Namespace) -> list[Config]:
    """Resolve the configs of every run the command line asks for.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options.

    Returns
    -------
    list of Config
        One validated config per run, in plan order.

    Raises
    ------
    SystemExit
        If ``--only`` names an entry that is not in the plan.
    """
    extra = command_line_layers(args)
    suffix = "_quick" if args.quick else ""
    if args.config:
        name = args.name or Path(args.config).stem
        return [resolve(args.config, extra, name=name + suffix)]

    common, entries = read_plan(args.plan)
    if args.only:
        missing = sorted(set(args.only) - {entry["name"] for entry in entries})
        if missing:
            sys.exit(f"not in {args.plan}: {missing}")
        entries = [entry for entry in entries if entry["name"] in args.only]
    else:
        entries = [entry for entry in entries if entry["enabled"]]
    return [resolve(entry["config"], [common, entry["set"], *extra], name=entry["name"] + suffix)
            for entry in entries]


def list_plan(path: Path) -> None:
    """Print the entries of the plan.

    Parameters
    ----------
    path : Path
        Plan file.
    """
    common, entries = read_plan(path)
    for entry in entries:
        cfg = resolve(entry["config"], [common, entry["set"]], name=entry["name"])
        state = "on " if entry["enabled"] else "off"
        print(f"[{state}] {entry['name']:<26} {cfg.description}")


def run(cfg: Config, pretrained_path: Path | None = None,
        checkpoint_path: Path | None = None) -> dict:
    """Run one experiment and write everything it produces to ``cfg.run_dir``.

    Parameters
    ----------
    cfg : Config
        Resolved settings of the run.
    pretrained_path : Path, optional
        Stage 1 weights to start from instead of pretraining.
    checkpoint_path : Path, optional
        Stage 2 weights to evaluate instead of training.

    Returns
    -------
    dict
        The content of ``metrics.json``.
    """
    device, run_dir = cfg.device, cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"\n=== {cfg.experiment} | device {device} | output {run_dir}")

    seed = cfg.seed + (cfg.split.fold + cfg.split.test_group if cfg.split.fold_seed else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    arrays, channels, coords = load_dataset(cfg)
    split = split_subjects(cfg, arrays)
    loaders = build_loaders(cfg, arrays, split)
    stage1_val = loaders.get("pretrain_val", loaders["val"])
    runs_of_epochs = cfg.data.context_epochs > 1
    n_channels = next(iter(arrays.values()))[0].shape[1]
    for name, subjects in split.items():
        print(f"{name:<6} {len(subjects):>2} subjects, {len(loaders[name].dataset):>6} windows",
              subjects if name != "train" else "")
    if cfg.data.scored_epochs:
        print(f"stage 1: {len(loaders['pretrain'].dataset)} training epochs "
              f"({len(loaders['pretrain'])} steps of {cfg.loader.batch_size}), "
              f"{len(loaders['pretrain_val'].dataset)} validation epochs; stage 2: "
              f"{len(loaders['train'])} steps of {cfg.finetune.batch_size or cfg.loader.batch_size}")

    model = GraphChangeNet(cfg, n_channels, coords).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_parameters:,}")

    pretrain_checkpoint = pretrained_path or run_dir / "pretrain_best.pt"
    if cfg.pretrain.init_run and not pretrained_path:
        pretrain_checkpoint = init_run_checkpoint(cfg)
    pretrain_history, history = [], []
    results: dict = {"experiment": cfg.experiment, "parameters": n_parameters,
                     "git": git_state(), "split": split}

    if checkpoint_path:
        load_weights(model, checkpoint_path, device)
    else:
        if pretrained_path:
            load_weights(model, pretrained_path, device)
        elif cfg.pretrain.init_run:
            load_pretrained_parts(model, pretrain_checkpoint, device)
            results["pretrained_from"] = str(pretrain_checkpoint)
            print(f"stage 1 weights from {pretrain_checkpoint}")
        elif cfg.pretrain.epochs:
            results["masked_loss_before"] = baselines(cfg, model, stage1_val)
            print("before:", results["masked_loss_before"])
            pretrain_history = pretrain(cfg, model, loaders.get("pretrain", loaders["train"]),
                                        stage1_val, pretrain_checkpoint,
                                        loaders.get("train_eval_pretrain"))
            results["masked_loss_after"] = baselines(cfg, model, stage1_val)
            print("after :", results["masked_loss_after"])
        history, results["val_best_macro_f1"] = classify(cfg, model, loaders["train"],
                                                         loaders["val"],
                                                         run_dir / "classify_best.pt",
                                                         loaders.get("train_eval"))
        probe = history[:cfg.finetune.freeze_epochs]
        if probe:
            results["val_probe_best_macro_f1"] = max(row["macro_f1"] for row in probe)

    true, predicted = predict(cfg, model, loaders["test"])
    results["test"] = scores(cfg, true, predicted)
    print("TEST:", {k: round(v, 4) for k, v in results["test"].items()})
    if cfg.data.scored_epochs:
        clean = loaders["test"].dataset.clean
        results["test_clean"] = {**scores(cfg, true[clean], predicted[clean]),
                                 "epochs": int(clean.sum()), "all_epochs": int(len(clean))}
        print(f"test, epochs below {cfg.data.artifact_uv:g} uV only "
              f"({results['test_clean']['epochs']} of {len(clean)}): kappa "
              f"{results['test_clean']['kappa']:.4f}, macro-F1 "
              f"{results['test_clean']['macro_f1']:.4f}")
    if runs_of_epochs:
        results["context"] = context_report(cfg, model, loaders["test"], results["test"])
        print(context_line(results["context"]))
        if cfg.pretrain.init_run:
            return finish(cfg, model, None, loaders, channels, coords, pretrain_history,
                          history, results, true, predicted)

    shuffled_mean, shuffled_std = evaluate_shuffled(cfg, model, loaders["test"])
    results["test_shuffled"] = {"mean": shuffled_mean, "std": shuffled_std,
                                "delta_kappa": shuffled_mean["kappa"] - results["test"]["kappa"],
                                "delta_macro_f1": shuffled_mean["macro_f1"]
                                - results["test"]["macro_f1"]}
    print(f"shuffled segments: delta kappa {results['test_shuffled']['delta_kappa']:+.4f}, "
          f"delta macro-F1 {results['test_shuffled']['delta_macro_f1']:+.4f}")
    if hasattr(model.sequence, "importance") and not runs_of_epochs:
        results["segment_graph"] = segment_graph_report(cfg, model, loaders["test"],
                                                        results["test"])
        print(segment_graph_line(results["segment_graph"]))
    if cfg.data.rolling:
        results["rolling"] = rolling_report(cfg, model, loaders)
        print(rolling_line(results["rolling"]))

    pretrained = None
    if Path(pretrain_checkpoint).exists():
        pretrained = GraphChangeNet(cfg, n_channels, coords).to(device)
        if cfg.pretrain.init_run and not pretrained_path:
            load_pretrained_parts(pretrained, pretrain_checkpoint, device)
        else:
            load_weights(pretrained, pretrain_checkpoint, device)
        results["reconstruction"] = reconstruction_report(cfg, pretrained, stage1_val, coords)
        print(format_reconstruction(results["reconstruction"]))
    return finish(cfg, model, pretrained, loaders, channels, coords, pretrain_history, history,
                  results, true, predicted)


def init_run_checkpoint(cfg: Config) -> Path:
    """The stage-1 weights that ``pretrain.init_run`` names, for the same fold.

    Parameters
    ----------
    cfg : Config
        Run settings.

    Returns
    -------
    Path
        ``<runs_dir>/<init_run>[_fold<k> | _rot<k>]/pretrain_best.pt``.

    Raises
    ------
    FileNotFoundError
        If that run has not finished stage 1.
    """
    name = cfg.pretrain.init_run
    if cfg.split.n_folds:
        name = f"{name}_fold{cfg.split.fold}"
    if cfg.split.n_test_groups:
        name = f"{name}_rot{cfg.split.test_group}"
    path = cfg.paths.runs_dir / name / "pretrain_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"pretrain.init_run: {path} does not exist; run "
                                f"{cfg.pretrain.init_run} first")
    return path


def finish(cfg: Config, model, pretrained, loaders: dict, channels: list[str],
           coords: np.ndarray, pretrain_history: list, history: list, results: dict,
           true: np.ndarray, predicted: np.ndarray) -> dict:
    """Write the metrics, histories, report and figures of a finished run.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : GraphChangeNet
        Network with its stage 2 weights.
    pretrained : GraphChangeNet or None
        Network with its stage 1 weights, if available.
    loaders : dict
        Loaders per split.
    channels : list of str
        Electrode names.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    pretrain_history, history : list
        Stage 1 and stage 2 histories.
    results : dict
        Everything measured; written to ``metrics.json``.
    true, predicted : np.ndarray
        Test stage indices.

    Returns
    -------
    dict
        ``results``.
    """
    run_dir = cfg.run_dir
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "history.json").write_text(json.dumps(
        {"pretrain": pretrain_history, "classify": history}, indent=2))
    report = [f"experiment : {cfg.experiment}", f"parameters : {results['parameters']:,}",
              "test       : " + ", ".join(f"{k}={v:.4f}" for k, v in results["test"].items())]
    if "test_clean" in results:
        report.append("test clean : " + ", ".join(
            f"{k}={v:.4f}" for k, v in results["test_clean"].items() if isinstance(v, float)))
    if "test_shuffled" in results:
        report.append(f"shuffled   : delta kappa {results['test_shuffled']['delta_kappa']:+.4f}")
    if "context" in results:
        report.append(context_line(results["context"]))
    if "segment_graph" in results:
        report.append(segment_graph_line(results["segment_graph"]))
    if "rolling" in results:
        report.append(rolling_line(results["rolling"]))
    if "reconstruction" in results:
        report.append(format_reconstruction(results["reconstruction"]))
    (run_dir / "report.txt").write_text("\n".join(report) + "\n")

    if cfg.evaluation.plots:
        write_figures(cfg, model, pretrained, loaders, channels, coords, pretrain_history,
                      history, results["test"], true, predicted)
    print(f"done: {run_dir}")
    return results


def context_line(report: dict) -> str:
    """One line on the epoch-context controls of Exp 14b.

    Parameters
    ----------
    report : dict
        Output of :func:`evaluate.context_report`.

    Returns
    -------
    str
        The drop in kappa and macro-F1 with the earlier epochs shuffled and
        with no earlier epochs.
    """
    return (f"context: shuffled earlier epochs delta kappa "
            f"{report['shuffled_context']['delta_kappa']:+.4f}, delta macro-F1 "
            f"{report['shuffled_context']['delta_macro_f1']:+.4f}; no earlier epochs delta "
            f"kappa {report['no_context']['delta_kappa']:+.4f}, delta macro-F1 "
            f"{report['no_context']['delta_macro_f1']:+.4f}")


def segment_graph_line(report: dict) -> str:
    """One line on the learned segment graph and its random-graph control.

    Parameters
    ----------
    report : dict
        Output of :func:`evaluate.segment_graph_report`.

    Returns
    -------
    str
        Edge shares and the drop in kappa and macro-F1 with random edges.
    """
    share = ", ".join(f"{name} {value:.3f}" for name, value in report["edge_share"].items())
    control = report["random_graph"]
    line = (f"segment graph: edges {share}; random graph delta kappa "
            f"{control['delta_kappa']:+.4f}, delta macro-F1 {control['delta_macro_f1']:+.4f}")
    if "no_graph_bias" in report:
        line += f"; no graph bias delta kappa {report['no_graph_bias']['delta_kappa']:+.4f}"
    if "mean_graph" in report:
        line += (f"; mean graph delta kappa {report['mean_graph']['delta_kappa']:+.4f}, raw "
                 f"similarity spread {report['raw_similarity']['spread']:.3f}")
    if "window_varying_share" in report:
        line += (f"; window-varying share {report['window_varying_share']:.3f}, stage F "
                 f"{report['stage_f']:.2f} (shuffled labels {report['stage_f_shuffled']:.2f})")
    return line


def rolling_line(report: dict) -> str:
    """One line on the extra scores of a rolling-window run.

    Parameters
    ----------
    report : dict
        Output of :func:`evaluate.rolling_report`.

    Returns
    -------
    str
        Kappa and macro-F1 per epoch and on the aligned windows, the drop
        without the elapsed time, and the elapsed-time-only baseline.
    """
    return (f"rolling: per 30 s epoch kappa {report['per_epoch']['kappa']:.4f} macro-F1 "
            f"{report['per_epoch']['macro_f1']:.4f}; aligned windows kappa "
            f"{report['aligned']['kappa']:.4f} macro-F1 {report['aligned']['macro_f1']:.4f}; "
            f"elapsed time zeroed delta kappa {report['no_elapsed']['delta_kappa']:+.4f}; "
            f"elapsed time only kappa {report['elapsed_only']['kappa']:.4f}")


def split_counts(loaders: dict) -> dict[str, tuple[int, int]]:
    """Subjects and items of every split, for the figure titles.

    Parameters
    ----------
    loaders : dict
        Loaders per split.

    Returns
    -------
    dict
        ``split -> (subjects, epochs)`` for ``pretrain``, ``train``, ``val``
        and ``test`` where present. Epochs are the labelled windows or epochs
        a split uses (the clean scored epochs in training, every scored epoch
        in validation and test), and for stage 1 every epoch it reads, so
        runs of epochs (Exp 14b+) are counted by their labelled epochs.
    """
    counts = {}
    for name in ("pretrain", "train", "val", "test"):
        if name not in loaders:
            continue
        dataset = loaders[name].dataset
        subjects = (len(dataset.epochs) if hasattr(dataset, "epochs")
                    else len(np.unique(getattr(dataset, "subject", [0]))))
        counts[name] = (subjects, len(getattr(dataset, "y", dataset)))
    return counts


def write_figures(cfg: Config, model, pretrained, loaders: dict, channels: list[str],
                  coords: np.ndarray, pretrain_history: list, history: list, test: dict,
                  true: np.ndarray, predicted: np.ndarray) -> None:
    """Write every figure of a run into ``cfg.run_dir``.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : GraphChangeNet
        Network with its stage 2 weights.
    pretrained : GraphChangeNet or None
        Network with its stage 1 weights, if available.
    loaders : dict
        Loaders per split.
    channels : list of str
        Electrode names.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    pretrain_history, history : list
        Stage 1 and stage 2 histories.
    test : dict
        Test metrics.
    true, predicted : np.ndarray
        Test stage indices.

    Runs of epochs that start from another run's stage 1 (Exp 14b) get the
    summary only: the graph and reconstruction figures belong to the run
    whose stage-1 weights they start from. Runs of epochs with their own
    stage 1 (Exp 15) draw them from the test epochs one by one
    (``test_single``).
    """
    import plots

    run_dir = cfg.run_dir
    plots.summary(cfg, pretrain_history, history, test, true, predicted, split_counts(loaders),
                  run_dir / "summary.png")
    if cfg.data.context_epochs > 1 and cfg.pretrain.init_run:
        return
    test_loader = loaders.get("test_single", loaders["test"])
    if pretrained is not None:
        plots.reconstruction_examples(cfg, pretrained, test_loader.dataset,
                                      run_dir / "reconstruction_examples.png")
    if hasattr(model.sequence, "importance"):
        segment = collect_segment_graphs(cfg, model, test_loader)
        np.savez_compressed(run_dir / "segment_graph.npz", **segment)
        plots.segment_graphs(cfg, segment, run_dir / "segment_graph.png")
    stable, deltas, stages = collect_graphs(cfg, model, test_loader)
    plots.scalp_graphs(cfg, stable, deltas, coords, run_dir / "graphs_scalp.png")
    graphs = plots.stage_graphs(cfg, stable, deltas, stages)
    plots.graphs_3d(cfg, graphs, channels, coords, run_dir / "graphs_3d.html")
    for index, name in enumerate(list(graphs)[:2]):
        plots.graph_projections(cfg, graphs[name], name, coords,
                                run_dir / f"graphs_projection_{index}.png")


FOLD_METRICS = ("kappa", "macro_f1", "accuracy", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_R")


def fold_summary(names: list[str]) -> None:
    """Print and save the mean and spread of k-fold runs over their folds.

    Every fold is scored on the same test subjects, so the spread shows how
    much the result depends on which subjects train and select the model.

    Parameters
    ----------
    names : list of str
        Run names without the ``_fold<k>`` suffix; their ``metrics.json`` are
        read from ``paths.runs_dir``, and the summary is written to
        ``<runs_dir>/<name>_kfold.json``.
    """
    runs_dir = resolve().paths.runs_dir
    for name in names:
        paths = sorted(runs_dir.glob(f"{name}_fold*/metrics.json"),
                       key=lambda path: int(path.parent.name.rsplit("_fold", 1)[1]))
        if not paths:
            print(f"{name}: no finished folds in {runs_dir}")
            continue
        rows = []
        for path in paths:
            result = json.loads(path.read_text())
            row = {"fold": path.parent.name.rsplit("_fold", 1)[1],
                   "val_macro_f1": result.get("val_best_macro_f1", float("nan")),
                   **{key: result["test"][key] for key in FOLD_METRICS}}
            if "test_shuffled" in result:
                row["shuffled_delta_kappa"] = result["test_shuffled"]["delta_kappa"]
            if "segment_graph" in result:
                row["random_graph_delta_kappa"] = result["segment_graph"]["random_graph"][
                    "delta_kappa"]
                row["positive_edge_share"] = result["segment_graph"]["edge_share"]["positive"]
                if "mean_abs_weight" in result["segment_graph"]:
                    row["mean_abs_edge_weight"] = result["segment_graph"]["mean_abs_weight"]
                if "no_graph_bias" in result["segment_graph"]:
                    row["no_graph_bias_delta_kappa"] = result["segment_graph"]["no_graph_bias"][
                        "delta_kappa"]
                for key in ("window_varying_share", "stage_f", "stage_f_shuffled"):
                    if key in result["segment_graph"]:
                        row[key] = result["segment_graph"][key]
                if "mean_graph" in result["segment_graph"]:
                    row["mean_graph_delta_kappa"] = result["segment_graph"]["mean_graph"][
                        "delta_kappa"]
                    row["raw_similarity_spread"] = result["segment_graph"]["raw_similarity"][
                        "spread"]
            if "rolling" in result:
                rolling = result["rolling"]
                row.update({"epoch_kappa": rolling["per_epoch"]["kappa"],
                            "epoch_macro_f1": rolling["per_epoch"]["macro_f1"],
                            "aligned_kappa": rolling["aligned"]["kappa"],
                            "aligned_macro_f1": rolling["aligned"]["macro_f1"],
                            "no_elapsed_delta_kappa": rolling["no_elapsed"]["delta_kappa"],
                            "elapsed_only_kappa": rolling["elapsed_only"]["kappa"]})
            if "val_probe_best_macro_f1" in result:
                row["probe_val_macro_f1"] = result["val_probe_best_macro_f1"]
            if "test_clean" in result:
                row["clean_kappa"] = result["test_clean"]["kappa"]
                row["clean_macro_f1"] = result["test_clean"]["macro_f1"]
            if "context" in result:
                row["shuffled_context_delta_kappa"] = result["context"]["shuffled_context"][
                    "delta_kappa"]
                row["no_context_delta_kappa"] = result["context"]["no_context"]["delta_kappa"]
            rows.append(row)
        keys = [key for key in rows[0] if key != "fold"]
        summary = {"experiment": name, "parameters": result["parameters"],
                   "folds": rows,
                   "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
                   "std": {key: float(np.std([row[key] for row in rows])) for key in keys}}
        (runs_dir / f"{name}_kfold.json").write_text(json.dumps(summary, indent=2))

        print(f"\n{name}: {len(rows)} folds, {result['parameters']:,} parameters")
        print(f"{'':<26}" + "".join(f"{'fold ' + row['fold']:>9}" for row in rows)
              + f"{'mean':>9}{'std':>9}")
        for key in keys:
            print(f"{key:<26}" + "".join(f"{row[key]:>9.4f}" for row in rows)
                  + f"{summary['mean'][key]:>9.4f}{summary['std'][key]:>9.4f}")


def main() -> None:
    """Resolve the requested runs and execute them one after another.

    A failing run is reported and the next one still starts; the exit code is
    non-zero if any run failed.
    """
    args = parse_args()
    if args.list:
        list_plan(args.plan)
        return
    if args.fold_summary:
        fold_summary(args.fold_summary)
        return

    configs = selected_configs(args)
    if not configs:
        sys.exit(f"nothing to run: no enabled entry in {args.plan} (use --only or --config)")
    if (args.pretrained or args.checkpoint) and len(configs) > 1:
        sys.exit("--pretrained / --checkpoint apply to a single run; add --only NAME or --config")
    if args.print_config:
        for cfg in configs:
            print(f"# {cfg.experiment}")
            print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return
    if args.prepare_data:
        for cfg in configs:
            arrays, _, _ = load_dataset(cfg)
            print(f"{cfg.experiment}: cache ready for {len(arrays)} subjects in "
                  f"{cfg.paths.cache_dir}")
        return

    finished, failed = [], []
    for cfg in configs:
        try:
            finished.append(run(cfg, args.pretrained, args.checkpoint))
        except Exception:
            traceback.print_exc()
            failed.append(cfg.experiment)

    print(f"\n{'experiment':<30}{'params':>11}{'kappa':>8}{'macro-F1':>10}{'acc':>8}")
    for result in finished:
        test = result["test"]
        print(f"{result['experiment']:<30}{result['parameters']:>11,}{test['kappa']:>8.4f}"
              f"{test['macro_f1']:>10.4f}{test['accuracy']:>8.4f}")
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
