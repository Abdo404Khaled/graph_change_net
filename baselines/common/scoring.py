from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import Config
from evaluate import scores


def save_predictions(path: Path, subject: np.ndarray, onset: np.ndarray, predicted: np.ndarray,
                     probability: np.ndarray) -> None:
    """Write one prediction per scored epoch, keyed by night and onset.

    Parameters
    ----------
    path : Path
        ``.npz`` file to write.
    subject : np.ndarray
        Subject id of each epoch.
    onset : np.ndarray
        Epoch onset in seconds, as in the hypnogram.
    predicted : np.ndarray
        Predicted stage index of each epoch.
    probability : np.ndarray
        Stage probabilities ``[n_epochs, n_stages]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, subject=np.asarray(subject, dtype=str), onset=np.asarray(onset, dtype=np.float64),
             predicted=np.asarray(predicted, dtype=np.int64),
             probability=np.asarray(probability, dtype=np.float32))


def align(table: dict[str, dict[str, np.ndarray]], predictions: dict[str, np.ndarray]
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match predictions to the reference epochs and check that they are the same set.

    Parameters
    ----------
    table : dict
        Output of :func:`baselines.common.reference.scored_epochs` for the test nights.
    predictions : dict
        Contents of a file written by :func:`save_predictions`.

    Returns
    -------
    true, predicted : np.ndarray
        Stage indices in reference order.
    peak : np.ndarray
        Largest absolute amplitude of each epoch in microvolts.
    subject : np.ndarray
        Subject id of each epoch.

    Raises
    ------
    ValueError
        If an epoch is predicted twice, missing, or not a scored test epoch.
    """
    keys = list(zip(predictions["subject"].tolist(), np.round(predictions["onset"], 3).tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("an epoch is predicted more than once")
    lookup = dict(zip(keys, predictions["predicted"].tolist()))
    true, predicted, peak, subject = [], [], [], []
    for name, epochs in table.items():
        for onset, stage, amplitude in zip(np.round(epochs["onset"], 3), epochs["stage"],
                                           epochs["peak"]):
            if (name, onset) not in lookup:
                raise ValueError(f"no prediction for {name} at {onset} s")
            true.append(int(stage))
            predicted.append(lookup.pop((name, onset)))
            peak.append(float(amplitude))
            subject.append(name)
    if lookup:
        raise ValueError(f"{len(lookup)} predictions are not scored test epochs, "
                         f"e.g. {next(iter(lookup))}")
    return np.array(true), np.array(predicted), np.array(peak), np.array(subject)


def score_run(cfg: Config, table: dict[str, dict[str, np.ndarray]], path: Path) -> dict:
    """Score a baseline's test predictions exactly as the reference model was scored.

    The metrics are those of :func:`evaluate.scores`, pooled over every scored
    test epoch; ``test_clean`` repeats them on the epochs at most
    ``data.artifact_uv``; ``per_subject`` gives each night's kappa and macro-F1.

    Parameters
    ----------
    cfg : Config
        Reference configuration (stage names, artifact threshold).
    table : dict
        Reference test epochs.
    path : Path
        Predictions file.

    Returns
    -------
    dict
        ``n_epochs``, ``test``, ``test_clean`` and ``per_subject``; also written
        next to the predictions as ``metrics.json``.
    """
    predictions = dict(np.load(path))
    true, predicted, peak, subject = align(table, predictions)
    clean = peak <= cfg.data.artifact_uv
    result = {"n_epochs": int(len(true)), "n_clean": int(clean.sum()),
              "test": scores(cfg, true, predicted),
              "test_clean": scores(cfg, true[clean], predicted[clean]),
              "per_subject": {name: {key: value for key, value in
                                     scores(cfg, true[subject == name],
                                            predicted[subject == name]).items()
                                     if key in ("kappa", "macro_f1")}
                              for name in table}}
    (path.parent / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
