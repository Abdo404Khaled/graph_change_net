"""Run Hypno-PC under its native configuration or ours on the fold's nights.

``native``: the authors' features from the EDFs, per-night z-scoring,
interpolation and clipping, then a pooled z-score and PCA fitted on the
training nights only, and their LinearSVC. ``ours``: the same EEG features
from exactly the tensors our model reads, a z-score fitted on the training
nights, the same PCA and LinearSVC, and the epoch with the 19 before it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import data
from baselines.common.native import night_epochs, subject_files
from baselines.common.ours import our_loaders, restrict_table
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import align, save_predictions, score_run
from baselines.hypnopc import features
from config import Config
from evaluate import scores

Nights = dict[str, dict[str, np.ndarray]]


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Extract or load the features, fit every row of the configuration and score it.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder; the primary row's predictions and ``metrics.json`` go
        here, the other rows' under ``rows/<name>``.
    quick : bool
        Smoke test: the first nights of each split (``settings["quick"]``) and
        the first ``quick_epochs`` epochs of each night.

    Returns
    -------
    dict
        Scores of the primary row, every row's summary, the parameter count
        and the timings.

    Raises
    ------
    ValueError
        If ``configuration`` is unknown, or the primary row's context differs
        from the reference run's.
    """
    started = time.time()
    cfg = load_reference_config(Path(settings["reference_run"]))
    split = load_fold(Path(settings["fold"]))
    if quick:
        split = {name: split[name][:settings["quick"][f"{name}_subjects"]] for name in split}
    limit = spec["quick_epochs"] if quick else None
    cache = (Path(settings["cache_dir"]) / "hypnopc"
             / (spec["configuration"] + ("_quick" if quick else "")))
    channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))

    if spec["configuration"] == "native":
        nights, names, seconds = native_features(cfg, split, spec, channels, cache, limit)
        design, front = native_design(nights, split, spec)
    elif spec["configuration"] == "ours":
        primary_context = spec["rows"][spec["primary_row"]]["context_epochs"]
        if primary_context != cfg.data.context_epochs:
            raise ValueError(f"primary row context {primary_context} differs from the "
                             f"reference run's {cfg.data.context_epochs}")
        nights, names, seconds = ours_features(cfg, split, spec, channels, cache, limit)
        design, front = ours_design(cfg, nights, split, spec)
    else:
        raise ValueError(f"unknown configuration {spec['configuration']!r}")

    table = scored_epochs(cfg, split["test"])
    rows = {}
    for row_name, row in spec["rows"].items():
        row_dir = out_dir if row_name == spec["primary_row"] else out_dir / "rows" / row_name
        rows[row_name] = fit_row(cfg, design, split, spec, row, front, table, row_dir, quick)

    primary = rows[spec["primary_row"]]
    result = {**primary["scores"],
              "parameters": primary["parameters"],
              "rows": {name: {key: value for key, value in row.items() if key != "scores"}
                       | {"test": row["scores"]["test"],
                          "per_subject": row["scores"]["per_subject"]}
                       for name, row in rows.items()},
              "n_features": len(names), "n_components": front["n_components"],
              "explained_variance": front["explained_variance"],
              "split": split, "quick": quick,
              "night_seconds": seconds, "total_seconds": time.time() - started}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def extract(jobs: int, cache: Path, tasks: dict[str, tuple]) -> dict[str, dict]:
    """Run per-night feature extraction in parallel, caching each night.

    Parameters
    ----------
    jobs : int
        Parallel worker processes.
    cache : Path
        Folder of the per-night ``.npz`` files.
    tasks : dict
        ``subject -> (cache_name, function, arguments)``.

    Returns
    -------
    dict
        ``subject -> {"features", "seconds"}``.
    """
    cache.mkdir(parents=True, exist_ok=True)
    todo = {s: task for s, task in tasks.items() if not (cache / task[0]).exists()}

    def build(subject: str, task: tuple) -> str:
        """Extract one night and write its cache file.

        Parameters
        ----------
        subject : str
            Subject id.
        task : tuple
            ``(cache_name, function, arguments)``.

        Returns
        -------
        str
            The subject id.
        """
        name, function, arguments = task
        values, seconds = function(*arguments)
        partial = cache / f"{name}.partial.npz"
        np.savez(partial, features=values, seconds=seconds)
        partial.replace(cache / name)
        return subject

    if todo:
        Parallel(n_jobs=min(jobs, len(todo)), backend="loky")(
            delayed(build)(subject, task) for subject, task in todo.items())
    loaded = {}
    for subject, task in tasks.items():
        stored = np.load(cache / task[0])
        loaded[subject] = {"features": stored["features"], "seconds": float(stored["seconds"])}
    return loaded


def native_features(cfg: Config, split: dict[str, list[str]], spec: dict, channels: list[str],
                    cache: Path, limit: int | None) -> tuple[Nights, list[str], dict]:
    """Native features of every hypnogram row of every split night.

    Epochs are cut at the onsets our pipeline keeps
    (:func:`baselines.common.native.night_epochs`), which are the upstream
    epochs joined to the hypnogram.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    split : dict
        Subject ids per split.
    spec : dict
        Native configuration.
    channels : list of str
        EEG electrode names.
    cache : Path
        Feature cache folder.
    limit : int or None
        Epochs per night (smoke tests), or all.

    Returns
    -------
    nights : dict
        ``subject -> {"features", "stage", "onset", "peak"}``.
    names : list of str
        Feature names.
    seconds : dict
        Extraction wall time per night.
    """
    files = subject_files(cfg)
    subjects = sum(split.values(), [])
    meta = {}
    for subject in subjects:
        epochs = night_epochs(cfg, subject)
        meta[subject] = {key: value[:limit] for key, value in epochs.items()}
    tasks = {s: (f"{s}.npz", features.native_night,
                 (str(files[s][0]), meta[s]["onset"], channels, spec)) for s in subjects}
    loaded = extract(spec["jobs"], cache, tasks)
    nights = {s: {"features": loaded[s]["features"], **meta[s]} for s in subjects}
    return (nights, features.native_feature_names(channels, spec),
            {s: loaded[s]["seconds"] for s in subjects})


def ours_features(cfg: Config, split: dict[str, list[str]], spec: dict, channels: list[str],
                  cache: Path, limit: int | None) -> tuple[Nights, list[str], dict]:
    """EEG features of every epoch of every split night, from our scaled data.

    The data and the robust scaler are those of :func:`baselines.common.ours.our_loaders`
    (:func:`data.build_epoch_loaders`, scaler fitted on the training nights).
    The cache file names carry a hash of the scaler, so a different fold
    never reads stale features.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    split : dict
        Subject ids per split.
    spec : dict
        Our-configuration settings.
    channels : list of str
        EEG electrode names.
    cache : Path
        Feature cache folder.
    limit : int or None
        Epochs per night (smoke tests), or all.

    Returns
    -------
    nights : dict
        ``subject -> {"features", "stage", "onset", "peak"}``.
    names : list of str
        Feature names.
    seconds : dict
        Extraction wall time per night.
    """
    loaders = our_loaders(cfg, split)
    datasets = {name: loaders[name].dataset for name in ("train", "val", "test")}
    center, scale = datasets["test"].center, datasets["test"].scale
    digest = hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()[:10]
    meta, tasks = {}, {}
    for name, dataset in datasets.items():
        for subject, epochs in zip(split[name], dataset.epochs):
            meta[subject] = {key: np.asarray(epochs[key])[:limit]
                             for key in ("stage", "onset", "peak")}
            tasks[subject] = (f"{subject}_{digest}.npz", features.ours_night,
                              (str(epochs["X"].filename), len(meta[subject]["stage"]),
                               center, scale, cfg.data.sfreq, spec))
    loaded = extract(spec["jobs"], cache, tasks)
    subjects = sum(split.values(), [])
    nights = {s: {"features": loaded[s]["features"], **meta[s]} for s in subjects}
    return (nights, features.spectral_feature_names(channels, spec["bands"]),
            {s: loaded[s]["seconds"] for s in subjects})


def night_zscore(values: np.ndarray, clip: float) -> np.ndarray:
    """Per-night z-score, linear interpolation and clipping, as upstream.

    Re-implements ``load_data_and_clean`` (``decomposition_utils.py:93-122``):
    mean and population SD over the night's scored epochs ignoring NaN, then
    ``DataFrame.interpolate("linear", axis=0)`` along the night, then
    clipping to ``[-clip, clip]``.

    Parameters
    ----------
    values : np.ndarray
        ``[n_scored_epochs, n_features]`` of one night, in time order.
    clip : float
        Clipping bound.

    Returns
    -------
    np.ndarray
        Normalised features; NaN only where no earlier value exists.
    """
    frame = pd.DataFrame(values)
    standardised = (frame - frame.mean()) / frame.std(ddof=0)
    return standardised.interpolate("linear", axis=0).clip(lower=-clip, upper=clip).to_numpy()


def native_design(nights: Nights, split: dict[str, list[str]], spec: dict
                  ) -> tuple[dict, dict]:
    """Single-epoch principal components of the scored epochs of every night (native).

    ``L`` and other unscored rows are dropped first (``decomposition_utils.py:77-80``);
    each night is z-scored by :func:`night_zscore`; the pooled z-score
    (sample SD, ``decomposition_utils.py:380-384``) and
    ``PCA(n_components=variance, random_state)`` (``:387-390``) are fitted on
    the training nights only and applied to every night.

    Parameters
    ----------
    nights : dict
        Output of :func:`native_features`.
    split : dict
        Subject ids per split.
    spec : dict
        Native configuration.

    Returns
    -------
    design : dict
        ``subject -> {"components", "stage", "onset", "train_rows", "eval_rows"}``.
    front : dict
        ``n_components``, ``explained_variance`` and ``parameters`` of the
        normalisation and PCA.

    Raises
    ------
    ValueError
        If a night still has missing values after interpolation.
    """
    normalised = {}
    for subject, night in nights.items():
        scored = night["stage"] >= 0
        values = night_zscore(night["features"][scored], spec["normalisation"]["clip"])
        if np.isnan(values).any():
            raise ValueError(f"{subject}: missing feature values remain after interpolation")
        normalised[subject] = {"values": values, "stage": night["stage"][scored],
                               "onset": night["onset"][scored]}
    pooled = np.concatenate([normalised[s]["values"] for s in split["train"]])
    mean, deviation = pooled.mean(axis=0), pooled.std(axis=0, ddof=1)
    pca = PCA(n_components=spec["pca"]["variance"], whiten=False,
              random_state=spec["pca"]["random_state"]).fit((pooled - mean) / deviation)
    design = {}
    for subject, night in normalised.items():
        rows = np.ones(len(night["stage"]), dtype=bool)
        design[subject] = {"components": pca.transform((night["values"] - mean) / deviation),
                           "stage": night["stage"], "onset": night["onset"],
                           "train_rows": rows, "eval_rows": rows}
    return design, front_summary(pca, pooled.shape[1])


def ours_design(cfg: Config, nights: Nights, split: dict[str, list[str]], spec: dict
                ) -> tuple[dict, dict]:
    """Single-epoch principal components of every epoch of every night (ours).

    The z-score and the PCA are fitted on the training nights' clean scored
    epochs (peak at most ``data.artifact_uv``, the epochs our model trains
    on) and applied to every epoch, lights-on and artefact epochs included,
    because they belong to the context. Missing values become the training
    mean (0) and values are clipped to ``[-clip, clip]``.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    nights : dict
        Output of :func:`ours_features`.
    split : dict
        Subject ids per split.
    spec : dict
        Our-configuration settings.

    Returns
    -------
    design : dict
        ``subject -> {"components", "stage", "onset", "train_rows", "eval_rows"}``.
    front : dict
        ``n_components``, ``explained_variance`` and ``parameters``.
    """
    clip = spec["normalisation"]["clip"]
    train_rows = {s: (n["stage"] >= 0) & (n["peak"] <= cfg.data.artifact_uv)
                  for s, n in nights.items()}
    pooled = np.concatenate([nights[s]["features"][train_rows[s]] for s in split["train"]])
    scaler = StandardScaler().fit(pooled)

    def normalise(values: np.ndarray) -> np.ndarray:
        """Training z-score, missing values to 0, clipping.

        Parameters
        ----------
        values : np.ndarray
            ``[n_epochs, n_features]``.

        Returns
        -------
        np.ndarray
            Normalised features.
        """
        return np.clip(np.nan_to_num(scaler.transform(values), nan=0.0), -clip, clip)

    pca = PCA(n_components=spec["pca"]["variance"], whiten=False,
              random_state=spec["pca"]["random_state"]).fit(normalise(pooled))
    design = {s: {"components": pca.transform(normalise(n["features"])), "stage": n["stage"],
                  "onset": n["onset"], "train_rows": train_rows[s], "eval_rows": n["stage"] >= 0}
              for s, n in nights.items()}
    return design, front_summary(pca, pooled.shape[1])


def front_summary(pca: PCA, n_features: int) -> dict:
    """Size of the normalisation and PCA in front of the classifier.

    Parameters
    ----------
    pca : PCA
        Fitted PCA.
    n_features : int
        Number of input features.

    Returns
    -------
    dict
        ``n_components``, ``explained_variance`` (total ratio) and
        ``parameters`` (z-score mean and SD, PCA mean and components).
    """
    return {"n_components": int(pca.n_components_),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "parameters": int(2 * n_features + pca.mean_.size + pca.components_.size)}


def stack_context(components: np.ndarray, context: int) -> np.ndarray:
    """Concatenate the components of each epoch and the ``context - 1`` before it.

    Before the first epoch of the night the first epoch is repeated, so no
    row uses a later epoch.

    Parameters
    ----------
    components : np.ndarray
        ``[n_epochs, K]`` in time order.
    context : int
        Epochs per row, the current one last.

    Returns
    -------
    np.ndarray
        ``[n_epochs, context * K]`` ordered ``t - context + 1, ..., t``.
    """
    index = np.arange(len(components))[:, None] + np.arange(1 - context, 1)[None, :]
    return components[np.maximum(index, 0)].reshape(len(components), -1)


def fit_row(cfg: Config, design: dict, split: dict[str, list[str]], spec: dict, row: dict,
            front: dict, table: dict, row_dir: Path, quick: bool) -> dict:
    """Fit one classifier row on the training nights and score it on val and test.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    design : dict
        Output of :func:`native_design` or :func:`ours_design`.
    split : dict
        Subject ids per split.
    spec : dict
        Configuration (``classifier`` settings).
    row : dict
        ``components`` (``all`` or a number), ``standardise_components`` and
        ``context_epochs``.
    front : dict
        Output of :func:`front_summary`.
    table : dict
        Reference test epochs (:func:`baselines.common.reference.scored_epochs`).
    row_dir : Path
        Folder for this row's predictions and ``metrics.json``.
    quick : bool
        Score only the predicted epochs (smoke tests).

    Returns
    -------
    dict
        ``scores`` (of :func:`baselines.common.scoring.score_run`), the
        paper-style ``per_subject_mean``, pooled ``val`` scores,
        ``components``, ``input_dimension``, ``train_epochs``, the solver's
        ``solver_iterations`` and whether it ``converged`` within
        ``classifier.max_iter``, and ``parameters``.
    """
    count = front["n_components"] if row["components"] == "all" else int(row["components"])
    selected = {s: d["components"][:, :count] for s, d in design.items()}
    component_parameters = 0
    if row["standardise_components"]:
        reference = np.concatenate([selected[s][design[s]["train_rows"]] for s in split["train"]])
        component_scaler = StandardScaler().fit(reference)
        selected = {s: component_scaler.transform(v) for s, v in selected.items()}
        component_parameters = 2 * count
    stacked = {s: stack_context(v, row["context_epochs"]) for s, v in selected.items()}

    train_x = np.concatenate([stacked[s][design[s]["train_rows"]] for s in split["train"]])
    train_y = np.concatenate([design[s]["stage"][design[s]["train_rows"]] for s in split["train"]])
    settings = spec["classifier"]
    model = LinearSVC(C=settings["C"], penalty=settings["penalty"], loss=settings["loss"],
                      dual=settings["dual"], tol=settings["tol"], max_iter=settings["max_iter"],
                      class_weight=settings["class_weight"],
                      random_state=settings["random_state"]).fit(train_x, train_y)

    def predict(subjects: list[str]) -> dict[str, np.ndarray]:
        """Predicted stage and pseudo-probabilities of every scored epoch of some nights.

        Parameters
        ----------
        subjects : list of str
            Night ids.

        Returns
        -------
        dict
            ``subject``, ``onset``, ``true``, ``predicted`` and ``probability``.
        """
        x = np.concatenate([stacked[s][design[s]["eval_rows"]] for s in subjects])
        decision = model.decision_function(x)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        logits = np.full((len(x), len(cfg.data.stages)), -np.inf)
        logits[:, model.classes_] = decision
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        return {"subject": np.concatenate([np.full(design[s]["eval_rows"].sum(), s)
                                           for s in subjects]),
                "onset": np.concatenate([design[s]["onset"][design[s]["eval_rows"]]
                                         for s in subjects]),
                "true": np.concatenate([design[s]["stage"][design[s]["eval_rows"]]
                                        for s in subjects]),
                "predicted": model.classes_[decision.argmax(axis=1)],
                "probability": probability}

    test = predict(split["test"])
    path = row_dir / "predictions.npz"
    save_predictions(path, test["subject"], test["onset"], test["predicted"], test["probability"])
    row_table = (restrict_table(table, test["subject"].tolist(), test["onset"].tolist())
                 if quick else table)
    scored = score_run(cfg, row_table, path)
    true, predicted, _, subject = align(row_table, dict(np.load(path)))
    per_subject = {name: {"kappa": float(cohen_kappa_score(true[subject == name],
                                                           predicted[subject == name])),
                          "accuracy": float(accuracy_score(true[subject == name],
                                                           predicted[subject == name]))}
                   for name in row_table}
    val = predict(split["val"]) if split["val"] else None
    return {"scores": scored,
            "per_subject_mean": {key: float(np.mean([v[key] for v in per_subject.values()]))
                                 for key in ("kappa", "accuracy")},
            "val": scores(cfg, val["true"], val["predicted"]) if val else None,
            "components": count, "context_epochs": row["context_epochs"],
            "standardise_components": row["standardise_components"],
            "input_dimension": int(train_x.shape[1]), "train_epochs": int(len(train_y)),
            "solver_iterations": int(model.n_iter_),
            "converged": bool(model.n_iter_ < settings["max_iter"]),
            "parameters": int(front["parameters"] + component_parameters + model.coef_.size
                              + model.intercept_.size)}
