from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data
from config import Config, load_run_config


def load_reference_config(run_dir: Path, overrides: dict | None = None) -> Config:
    """The exact configuration of the reference run, as it was saved.

    The saved ``config.yaml`` is already fully resolved, so it is built
    directly; the experiment chain in ``configs/`` is not read again.

    Parameters
    ----------
    run_dir : Path
        Run directory of the reference experiment.
    overrides : dict, optional
        Nested settings merged on top, with :func:`config.merge`.

    Returns
    -------
    Config
        The run's configuration.
    """
    return load_run_config(run_dir, overrides or {})


def load_fold(path: Path) -> dict[str, list[str]]:
    """Subject ids of one fold.

    Parameters
    ----------
    path : Path
        Fold file written by :func:`write_fold`.

    Returns
    -------
    dict
        ``train``, ``val`` and ``test`` subject ids.
    """
    fold = json.loads(path.read_text())
    return {name: list(fold[name]) for name in ("train", "val", "test")}


def write_fold(run_dir: Path, path: Path) -> dict:
    """Copy the split a reference run used into a fold file every baseline reads.

    The split is copied from the run's ``metrics.json`` rather than drawn
    again, so the baselines train, select and test on exactly the subjects
    the reference model did.

    Parameters
    ----------
    run_dir : Path
        Run directory of the reference experiment.
    path : Path
        Fold file to write.

    Returns
    -------
    dict
        The fold, with its source and the reference run's commit.
    """
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fold = {"source": str(run_dir / "metrics.json"), "experiment": metrics["experiment"],
            "commit": metrics["git"]["commit"], **metrics["split"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fold, indent=1) + "\n")
    return fold


def scored_epochs(cfg: Config, subjects: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Every scored epoch of the given nights, as our pipeline defines them.

    Reads only the cached epoch metadata (stage, onset, peak) that
    :func:`data.load_subject_epochs` wrote, so the set of epochs and their
    labels are exactly the ones the reference model was scored on.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    subjects : list of str
        Subject ids.

    Returns
    -------
    dict
        ``subject -> {"onset", "stage", "peak"}`` for the scored epochs only.

    Raises
    ------
    FileNotFoundError
        If a subject has no cached epochs for this configuration.
    """
    channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
    signature = data.preprocessing_signature(cfg, channels)
    table = {}
    for subject in subjects:
        meta_path = cfg.paths.cache_dir / f"{subject}_{signature}_epochs_meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} is missing; run the reference pipeline first")
        meta = np.load(meta_path)
        scored = meta["stage"] >= 0
        table[subject] = {key: meta[key][scored] for key in ("onset", "stage", "peak")}
    return table
