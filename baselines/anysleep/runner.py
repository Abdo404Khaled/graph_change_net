"""AnySleep under its released configuration (``native``) and under ours (``ours``).

``native`` runs the released checkpoints zero-shot, preprocessed as
``upstream/examples/anphy/predict_anphy.py`` does. ``ours`` trains the same
architecture from scratch on our data protocol with our stage-2 trainer. See
the README for every decision.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

from baselines.anysleep.model import AnySleepRuns, released_network, upstream_module
from baselines.common.native import cached_night, read_channels, subject_files
from baselines.common.ours import run_ours
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run

HERE = Path(__file__).resolve().parent


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Run one AnySleep configuration and score it on the fold's test nights.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test on fewer nights, items and epochs.

    Returns
    -------
    dict
        The scores.

    Raises
    ------
    ValueError
        If the configuration names an unknown mode.
    """
    if spec["mode"] == "native":
        return run_native(spec, settings, out_dir, quick)
    if spec["mode"] == "ours":
        return run_our_configuration(spec, settings, out_dir, quick)
    raise ValueError(f"unknown mode {spec['mode']!r}")


def cache_folder(spec: dict, settings: dict, quick: bool) -> Path:
    """Cache folder of one configuration, separate for smoke tests.

    Parameters
    ----------
    spec : dict
        The configuration (``name``).
    settings : dict
        Shared settings (``cache_dir``).
    quick : bool
        Smoke test.

    Returns
    -------
    Path
        ``<cache_dir>/anysleep/<name>[_quick]``.
    """
    return Path(settings["cache_dir"]) / "anysleep" / (spec["name"] + ("_quick" if quick else ""))


def preprocessing_key(spec: dict) -> str:
    """Short hash of every setting that changes a preprocessed night.

    Parameters
    ----------
    spec : dict
        Native configuration.

    Returns
    -------
    str
        Ten hexadecimal characters.
    """
    content = json.dumps({"channels": spec["channels"], **spec["preprocessing"]}, sort_keys=True)
    return hashlib.sha1(content.encode()).hexdigest()[:10]


def native_night(spec: dict, edf_path: Path) -> np.ndarray:
    """One night preprocessed exactly as ``predict_anphy.py`` preprocesses it.

    ``predict_anphy.preprocess`` resamples all channels with
    ``resample_poly(data, 128, sfreq, axis=1)``, clips each channel to
    ``+-20 * IQR`` with ``clip_noisy_values`` and scales each channel with a
    ``RobustScaler`` fitted on that night. Resampling is applied channel by
    channel here, which gives the same result because the polyphase filter
    acts on each row separately, and keeps the night from being held at
    1 kHz for every channel at once. The upstream clip function is called
    with the configured multiple. The signals are in microvolts rather than
    volts; clipping and scaling are both relative to each channel's IQR, so
    the result does not depend on the unit.

    Parameters
    ----------
    spec : dict
        Native configuration (``channels``, ``preprocessing``).
    edf_path : Path
        EDF recording.

    Returns
    -------
    np.ndarray
        ``[n_channels, samples]`` float32 at the configured sampling rate.
    """
    preprocessing = spec["preprocessing"]
    upstream = upstream_module("predict_anphy")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    source_rate = int(raw.info["sfreq"])
    raw.close()
    target_rate = int(preprocessing["sampling_rate_hz"])
    signals = read_channels(edf_path, spec["channels"],
                            lambda x: resample_poly(x, target_rate, source_rate)
                            if source_rate != target_rate else x).astype(np.float64)
    signals = upstream.clip_noisy_values(
        signals, min_max_times_global_iqr=preprocessing["clip_iqr_multiple"])
    return RobustScaler().fit_transform(signals.T).T.astype(np.float32)


def onset_rows(onsets: np.ndarray, epoch_seconds: float, n_outputs: int, subject: str
               ) -> np.ndarray:
    """Model output index of each epoch onset.

    The model's ``k``-th output pools samples ``[k, k + 1)`` epochs from the
    start of the recording, so an epoch starting at ``onset`` seconds is
    output ``onset / epoch_seconds``.

    Parameters
    ----------
    onsets : np.ndarray
        Epoch onsets in seconds from the start of the recording.
    epoch_seconds : float
        Epoch length.
    n_outputs : int
        Outputs the model gave for the night.
    subject : str
        Subject id, for the error message.

    Returns
    -------
    np.ndarray
        Output index of each onset.

    Raises
    ------
    ValueError
        If an onset is not on the model's epoch grid or past its last output.
    """
    position = onsets / epoch_seconds
    rows = np.round(position).astype(np.int64)
    if not np.allclose(position, rows):
        raise ValueError(f"{subject}: onsets are not multiples of {epoch_seconds} s")
    if rows.min() < 0 or rows.max() >= n_outputs:
        raise ValueError(f"{subject}: epoch {rows.max()} has no output (only {n_outputs})")
    return rows


def summarise(runs: dict[str, dict]) -> tuple[dict, dict]:
    """Mean and standard deviation of every test score over the checkpoints.

    The standard deviation is the population one (``np.std``), as in
    ``upstream/scripts/final-figures/table_s5.ipynb``.

    Parameters
    ----------
    runs : dict
        ``checkpoint -> scores`` from :func:`baselines.common.scoring.score_run`.

    Returns
    -------
    mean, deviation : dict
        Per test score.
    """
    keys = next(iter(runs.values()))["test"].keys()
    values = {key: np.array([run["test"][key] for run in runs.values()]) for key in keys}
    return ({key: float(value.mean()) for key, value in values.items()},
            {key: float(value.std()) for key, value in values.items()})


def causal_softmax(model: torch.nn.Module, inputs: torch.Tensor, rows: np.ndarray,
                   window_epochs: int, epoch_samples: int, batch: int) -> np.ndarray:
    """Softmax of every scored epoch from a window that ends at that epoch.

    The released model is fully convolutional, and the native run passes the
    whole night at once, so every epoch sees later epochs. Here epoch ``r`` is
    predicted from epochs ``r - window_epochs + 1 .. r`` (the training sequence
    length) and only the last output is kept, so no later epoch reaches it.
    AnySleep was trained with the target epoch at a uniformly random position of
    its window, the last one included. The first ``window_epochs - 1`` epochs of
    a night use the history they have.

    Parameters
    ----------
    model : torch.nn.Module
        Released network in evaluation mode.
    inputs : torch.Tensor
        ``[1, samples, channels]`` preprocessed night on the model's device.
    rows : np.ndarray
        Output index (epoch) of each scored epoch.
    window_epochs : int
        Epochs per window.
    epoch_samples : int
        Samples per epoch at the model's rate.
    batch : int
        Full-length windows per forward pass.

    Returns
    -------
    np.ndarray
        ``[len(rows), 5]`` softmax.
    """
    out = np.zeros((len(rows), 5), dtype=np.float32)
    full = [i for i, r in enumerate(rows) if r >= window_epochs - 1]
    short = [i for i, r in enumerate(rows) if r < window_epochs - 1]
    with torch.no_grad():
        for first in range(0, len(full), batch):
            chunk = full[first:first + batch]
            x = torch.cat([inputs[:, (rows[i] - window_epochs + 1) * epoch_samples:
                                  (rows[i] + 1) * epoch_samples] for i in chunk])
            out[chunk] = model(x)[:, -1].softmax(-1).float().cpu().numpy()
        for i in short:
            x = inputs[:, :(rows[i] + 1) * epoch_samples]
            out[i] = model(x)[0, -1].softmax(-1).float().cpu().numpy()
    return out


def run_native(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """The released checkpoints, zero-shot, on whole test nights.

    With ``causal_window_epochs`` in the spec (``causal.yaml``) every scored
    epoch is instead predicted from the window that ends at it
    (:func:`causal_softmax`); everything else is unchanged.

    Each night is preprocessed once (:func:`native_night`, cached), passed
    whole through each checkpoint in one forward pass, and every scored
    epoch is given the output at its onset. Each checkpoint is scored on its
    own and the scores are summarised as mean and standard deviation over
    the checkpoints, as the paper reports them; the average of the three
    checkpoints' softmax outputs is scored as an extra row.

    Parameters
    ----------
    spec : dict
        Native configuration.
    settings : dict
        Shared settings.
    out_dir : Path
        Output folder.
    quick : bool
        Score only the first ``quick.test_subjects`` test nights.

    Returns
    -------
    dict
        ``test`` (mean over checkpoints), ``test_sd``, ``runs``, ``ensemble``,
        ``n_epochs``, ``parameters`` and timings.
    """
    cfg = load_reference_config(Path(settings["reference_run"]))
    split = load_fold(Path(settings["fold"]))
    subjects = split["test"][:settings["quick"]["test_subjects"]] if quick else split["test"]
    table = scored_epochs(cfg, subjects)
    files = subject_files(cfg)
    cache = cache_folder(spec, settings, quick)
    key = preprocessing_key(spec)
    epoch_seconds = spec["preprocessing"]["epoch_seconds"]
    device = cfg.device
    checkpoints = {Path(path).stem: HERE / path for path in spec["checkpoints"]}

    probabilities = {name: {} for name in checkpoints}
    timings = {}
    for subject in subjects:
        started = time.perf_counter()
        night = cached_night(cache / f"{subject}_{key}.npy",
                             lambda subject=subject: native_night(spec, files[subject][0]))
        prepared = time.perf_counter()
        inputs = torch.from_numpy(np.ascontiguousarray(night.T)).unsqueeze(0).to(device)
        rows = None
        for name, checkpoint in checkpoints.items():
            model = released_network(checkpoint, spec["network"]).to(device).eval()
            if "causal_window_epochs" in spec:
                epoch_samples = int(round(spec["preprocessing"]["sampling_rate_hz"] * epoch_seconds))
                rows = onset_rows(table[subject]["onset"], epoch_seconds,
                                  night.shape[1] // epoch_samples, subject)
                probabilities[name][subject] = causal_softmax(
                    model, inputs, rows, spec["causal_window_epochs"], epoch_samples,
                    spec["causal_batch"])
            else:
                with torch.no_grad():
                    logits = model(inputs)[0]
                if rows is None:
                    rows = onset_rows(table[subject]["onset"], epoch_seconds, logits.shape[0],
                                      subject)
                probabilities[name][subject] = logits.softmax(-1).cpu().numpy()[rows]
            del model
        del inputs
        timings[subject] = {"preprocess_s": prepared - started,
                            "inference_s": time.perf_counter() - prepared,
                            "channels": int(night.shape[0]), "samples": int(night.shape[1])}
        torch.cuda.empty_cache()

    names = np.concatenate([np.full(len(table[s]["onset"]), s) for s in subjects])
    onsets = np.concatenate([table[s]["onset"] for s in subjects])
    runs = {}
    for name in checkpoints:
        probability = np.concatenate([probabilities[name][s] for s in subjects])
        path = out_dir / name / "predictions.npz"
        save_predictions(path, names, onsets, probability.argmax(-1), probability)
        runs[name] = score_run(cfg, table, path)
    ensemble = np.mean([np.concatenate([probabilities[name][s] for s in subjects])
                        for name in checkpoints], axis=0)
    path = out_dir / "ensemble" / "predictions.npz"
    save_predictions(path, names, onsets, ensemble.argmax(-1), ensemble)
    ensemble_scores = score_run(cfg, table, path)

    mean, deviation = summarise(runs)
    clean_mean, clean_deviation = summarise({name: {"test": run["test_clean"]}
                                             for name, run in runs.items()})
    parameters = sum(p.numel() for p in
                     released_network(next(iter(checkpoints.values())), spec["network"])
                     .parameters())
    result = {"n_epochs": runs[next(iter(runs))]["n_epochs"], "subjects": subjects,
              "test": mean, "test_sd": deviation, "test_clean": clean_mean,
              "test_clean_sd": clean_deviation, "runs": runs, "ensemble": ensemble_scores,
              "parameters": parameters, "quick": quick, "timing": timings,
              "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
              if torch.cuda.is_available() else None}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def run_our_configuration(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """The AnySleep architecture trained from scratch under our configuration.

    The network is built after seeding with ``spec["seed"]``, then trained
    and scored by :func:`baselines.common.ours.run_ours`.

    Parameters
    ----------
    spec : dict
        ``ours`` configuration (``seed``, ``network``, ``overrides``).
    settings : dict
        Shared settings.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        The scores of :func:`run_ours`, with timing and peak GPU memory.

    Raises
    ------
    ValueError
        If the pooling width is not one epoch of our data.
    """
    cfg = load_reference_config(Path(settings["reference_run"]))
    epoch_samples = int(round(cfg.data.sfreq * cfg.data.window_s))
    if spec["network"]["pool_samples"] != epoch_samples:
        raise ValueError(f"pool_samples is {spec['network']['pool_samples']}, "
                         f"but an epoch has {epoch_samples} samples")
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    network = AnySleepRuns(**spec["network"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = run_ours(network, settings, spec["overrides"], out_dir, quick)
    result["seconds"] = time.perf_counter() - started
    result["max_memory_allocated_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                                         if torch.cuda.is_available() else None)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result
