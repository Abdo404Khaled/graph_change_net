"""The native configuration of the sleep-graph-benchmark models, run on our fold and scored by us.

Upstream's own code does everything that defines the configuration: ANPHY
preparation (``preprocessing.prepare``), training and model selection
(``training.train``). Our side writes the fold in upstream's split format,
predicts every scored test epoch from upstream's selected checkpoint
(:mod:`baselines.sleepgraph.native_predict`, see there why), checks that
upstream's epochs and stages are ours, and scores with our scorer. Upstream
runs in separate processes with its repository first on ``sys.path``,
because its ``data`` package would collide with our ``data`` module.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from config import Config

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
REPOSITORY = ROOT.parents[1]


def upstream_environment() -> dict[str, str]:
    """The environment of an upstream process: its repository first on the path.

    Returns
    -------
    dict
        ``os.environ`` with ``PYTHONPATH`` starting at ``upstream/``.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(UPSTREAM)] + [p for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p])
    return environment


def call(arguments: list[str]) -> None:
    """Run an upstream command in its repository and fail loudly.

    Parameters
    ----------
    arguments : list of str
        Command line after the interpreter.

    Raises
    ------
    RuntimeError
        If the command exits with an error.
    """
    command = [sys.executable] + arguments
    print("upstream:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=UPSTREAM, env=upstream_environment())
    if result.returncode:
        raise RuntimeError(f"upstream command failed ({result.returncode}): {' '.join(command)}")


def subjects_of(split: dict[str, list[str]]) -> list[str]:
    """Every subject of a split, in split order.

    Parameters
    ----------
    split : dict
        Subject ids per split.

    Returns
    -------
    list of str
        Train, validation and test subjects.
    """
    return split["train"] + split["val"] + split["test"]


def prepare(cfg: Config, subjects: list[str], folder: Path) -> Path:
    """Upstream's ANPHY preparation of the given nights, once.

    Each night's raw folder is linked into ``folder/raw`` and upstream's
    ``preprocessing.prepare`` writes ``folder/prepared/<subject>.npz`` with
    its defaults (19 EEG + 2 EOG + 2 EMG channels, its filters, 100 Hz).
    Nights already prepared are skipped by upstream.

    Parameters
    ----------
    cfg : Config
        Reference configuration (``paths.data_root``).
    subjects : list of str
        Nights to prepare; the prepared folder must hold exactly these,
        because upstream's trainers load every file in it.
    folder : Path
        Cache folder of this preparation.

    Returns
    -------
    Path
        The prepared folder.

    Raises
    ------
    RuntimeError
        If a night is missing after preparation, or other nights are present.
    """
    raw, prepared = folder / "raw", folder / "prepared"
    raw.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        link = raw / subject
        if not link.exists():
            link.symlink_to((REPOSITORY / cfg.paths.data_root / subject).resolve())
    missing = [s for s in subjects if not (prepared / f"{s}.npz").exists()]
    if missing:
        call(["-m", "preprocessing.prepare", "--dataset", "anphy", "--raw-dir", str(raw),
              "--output-dir", str(prepared)])
    present = sorted(path.stem for path in prepared.glob("*.npz"))
    if present != sorted(subjects):
        raise RuntimeError(f"{prepared} holds {present}, expected {sorted(subjects)}")
    return prepared


def write_split(split: dict[str, list[str]], path: Path) -> Path:
    """Our fold in upstream's split format, as its fold 0.

    Parameters
    ----------
    split : dict
        Subject ids per split.
    path : Path
        Output JSON.

    Returns
    -------
    Path
        ``path``.
    """
    fold = {f"{name}_files": [f"{s}.npz" for s in split[name]] for name in ("train", "val", "test")}
    fold.update({f"{name}_subjects": split[name] for name in ("train", "val", "test")})
    payload = {"schema_version": 1, "dataset": "ANPHY", "n_splits": 1,
               "strategy": "baselines/folds (our fold, written by baselines.sleepgraph.native)",
               "folds": {"0": fold}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def check_epochs(cfg: Config, predictions: dict[str, np.ndarray], subjects: list[str]) -> dict:
    """Check that upstream's test epochs and stages are exactly our scored epochs.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    predictions : dict
        Output of :mod:`native_predict` (``recording_id``, ``epoch_index``,
        ``label``).
    subjects : list of str
        Test subjects.

    Returns
    -------
    dict
        Counts of matched epochs per subject.

    Raises
    ------
    ValueError
        If an epoch is missing on either side or a stage differs.
    """
    table = scored_epochs(cfg, subjects)
    counts = {}
    for subject in subjects:
        ours = dict(zip(np.round(table[subject]["onset"], 3), table[subject]["stage"]))
        rows = predictions["recording_id"] == subject
        theirs = dict(zip(np.round(predictions["epoch_index"][rows] * 30.0, 3),
                          predictions["label"][rows]))
        if set(ours) != set(theirs):
            raise ValueError(f"{subject}: {len(set(ours) - set(theirs))} of our epochs missing, "
                             f"{len(set(theirs) - set(ours))} extra in upstream's")
        wrong = sum(int(ours[k]) != int(theirs[k]) for k in ours)
        if wrong:
            raise ValueError(f"{subject}: {wrong} epochs with a different stage")
        counts[subject] = len(ours)
    return counts


def run_native(kind: str, spec: dict, settings: dict, out_dir: Path, quick: bool,
               launcher: Path | None = None, distance: Path | None = None) -> dict:
    """Prepare, train with upstream's trainer, predict every test epoch, score.

    Parameters
    ----------
    kind : str
        Model folder name (``graphsleepnet``, ``graphsleepnet_distance``,
        ``mstgcn``, ``productgraphsleepnet``).
    spec : dict
        Contents of the model's ``native.yaml`` or ``cnn2s.yaml``: ``model``
        (upstream's ``--model``), ``arguments`` (extra upstream arguments,
        normally none), ``quick_arguments`` and ``predict_batch``; a
        ``cnn2s.yaml`` also names upstream's ``condition`` (``cnn_2s``) and
        the predictor's ``predict_kind``; a ``causal.yaml`` sets ``causal``,
        which trains through ``causal_launcher.py`` and predicts with
        past-only windows. Without these keys the run is upstream's
        ``native`` condition predicted as ``kind``, as before.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test on ``quick.native_nights`` training nights and the quick
        numbers of validation and test nights, with ``quick_arguments``.
    launcher : Path, optional
        Script run instead of ``-m training.train`` (the distance variant).
    distance : Path, optional
        Fixed adjacency file given to the predictor.

    Returns
    -------
    dict
        Scores, matched epochs, agreement with upstream's own predictions.
    """
    out_dir = out_dir.resolve()
    cfg = load_reference_config(Path(settings["reference_run"]))
    split = load_fold(Path(settings["fold"]))
    if quick:
        quick_settings = settings["quick"]
        split = {"train": split["train"][:quick_settings["native_nights"]],
                 "val": split["val"][:quick_settings["val_subjects"]],
                 "test": split["test"][:quick_settings["test_subjects"]]}
    cache = REPOSITORY / settings["cache_dir"] / "sleepgraph" / ("native_quick" if quick
                                                                 else "native")
    prepared = prepare(cfg, subjects_of(split), cache)
    split_file = write_split(split, out_dir / "split.json")
    run_dir = out_dir / "upstream"
    arguments = ["--dataset", "anphy", "--model", spec["model"],
                 "--condition", spec.get("condition", "native"),
                 "--fold-id", "0", "--split-file", str(split_file), "--data-dir", str(prepared),
                 "--output-dir", str(run_dir)] + [str(a) for a in spec["arguments"]]
    if quick:
        arguments += [str(a) for a in spec["quick_arguments"]]
    if spec.get("causal"):
        launcher = ROOT / "causal_launcher.py"
    call(([str(launcher)] if launcher else ["-m", "training.train"]) + arguments)

    predicted_path = out_dir / "native_predictions.npz"
    predict = [str(ROOT / "native_predict.py"), "--kind", spec.get("predict_kind", kind),
               "--run-dir", str(run_dir),
               "--data-dir", str(prepared), "--split-file", str(split_file),
               "--out", str(predicted_path), "--batch-size", str(spec["predict_batch"])]
    if distance is not None:
        predict += ["--distance", str(distance.resolve())]
    if spec.get("causal"):
        predict.append("--causal")
    call(predict)
    predicted = dict(np.load(predicted_path, allow_pickle=True))
    counts = check_epochs(cfg, predicted, split["test"])
    onset = predicted["epoch_index"] * 30.0
    path = out_dir / "predictions.npz"
    save_predictions(path, predicted["recording_id"], onset,
                     predicted["probability"].argmax(1), predicted["probability"])
    np.savez_compressed(out_dir / "graphs.npz", subject=predicted["recording_id"], onset=onset,
                        graph=predicted["graph"], channels=predicted["channels"])
    result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
    upstream_metrics = json.loads((run_dir / "fold_0" / "test_metrics.json").read_text())
    summary = json.loads(str(predicted["summary"]))
    if summary["agreement_with_upstream"] is not None and summary["agreement_with_upstream"] < 1:
        print(f"warning: {summary['agreement_with_upstream']:.4f} of the windows upstream also "
              "predicted agree with its own predictions", flush=True)
    result.update({"configuration": spec["configuration"], "quick": quick, "split": split,
                   "parameters": summary["parameters"],
                   "agreement_with_upstream": summary["agreement_with_upstream"],
                   "upstream_windows": summary["upstream_epochs"],
                   "matched_epochs": counts, "channels": [str(c) for c in predicted["channels"]],
                   "upstream_test_metrics": upstream_metrics})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def main() -> None:
    """Command line: ``python -m baselines.sleepgraph.native``, upstream's preparation of every fold night.

    The four native graph baselines share one prepared folder; preparing it
    once before their jobs start keeps them from writing the same files at
    the same time.
    """
    settings = yaml.safe_load((ROOT.parent / "settings.yaml").read_text())
    cfg = load_reference_config(Path(settings["reference_run"]))
    split = load_fold(Path(settings["fold"]))
    prepared = prepare(cfg, subjects_of(split),
                       REPOSITORY / settings["cache_dir"] / "sleepgraph" / "native")
    print(f"prepared {len(list(prepared.glob('*.npz')))} nights in {prepared}")


if __name__ == "__main__":
    main()
