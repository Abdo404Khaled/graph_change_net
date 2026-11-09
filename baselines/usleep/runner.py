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

import data
from baselines.common.native import cached_night, night_epochs, read_channels, subject_files
from baselines.common.ours import run_ours
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from baselines.usleep.model import USleepNetwork, pretrained_usleep
from config import Config


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Run U-Sleep under the configuration ``spec`` describes.

    Parameters
    ----------
    spec : dict
        Contents of ``baselines/usleep/native.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        The run's scores.

    Raises
    ------
    ValueError
        If ``spec["protocol"]`` is unknown.
    """
    if spec["protocol"] == "native_zero_shot":
        return run_native(spec, settings, out_dir, quick)
    if spec["protocol"] == "ours":
        return run_our_configuration(spec, settings, out_dir, quick)
    raise ValueError(f"unknown protocol {spec['protocol']!r}")


def run_our_configuration(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Train U-Sleep from scratch on our data protocol and score it.

    The architecture is the released one except where our input forces a
    change (input channels, and the epoch length in samples for the
    segment classifier's pooling), as ``spec["architecture"]`` sets it.

    Parameters
    ----------
    spec : dict
        ``seed``, ``architecture`` (arguments of :class:`USleepNetwork`) and
        ``overrides`` (merged on the reference configuration).
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        The scores of :func:`baselines.common.ours.run_ours`.
    """
    torch.manual_seed(spec["seed"])
    network = USleepNetwork(**spec["architecture"])
    return run_ours(network, settings, spec["overrides"], out_dir, quick)


def eeg_eog_pairs(cfg: Config, spec: dict, quick: bool) -> list[tuple[str, str]]:
    """Every (EEG, EOG) channel pair U-Sleep is run on, as AnySleep ran it on ANPHY.

    The EEG channels are our montage's electrodes minus ``spec["excluded_eeg"]``;
    the EOG channels are ``spec["eog_channels"]``.

    Parameters
    ----------
    cfg : Config
        Reference configuration (``paths.data_root``).
    spec : dict
        Native configuration.
    quick : bool
        Keep only the first ``spec["quick"]["pairs"]`` pairs.

    Returns
    -------
    list of tuple
        ``(eeg, eog)`` names.
    """
    montage, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
    excluded = {name.upper() for name in spec["excluded_eeg"]}
    eeg = [name for name in montage if name.upper() not in excluded]
    pairs = [(first, second) for first in eeg for second in spec["eog_channels"]]
    return pairs[:spec["quick"]["pairs"]] if quick else pairs


def native_transform(spec: dict, source_rate: int):
    """U-Sleep's per-channel preprocessing of a whole night, in AnySleep's order.

    Polyphase resampling to ``spec["sampling_rate"]``, clipping to
    +-``spec["clip_iqr"]`` times the channel's interquartile range (centred
    on 0, as ``clip_noisy_values`` does), then scikit-learn's
    ``RobustScaler`` fitted on the night itself. No filter and no
    re-referencing.

    Parameters
    ----------
    spec : dict
        Native configuration.
    source_rate : int
        Sampling rate of the recording.

    Returns
    -------
    callable
        Maps a 1-D microvolt signal to the preprocessed float32 signal.
    """
    def transform(x: np.ndarray) -> np.ndarray:
        """Resample, clip and robust-scale one channel.

        Parameters
        ----------
        x : np.ndarray
            1-D signal at ``source_rate``.

        Returns
        -------
        np.ndarray
            1-D float32 signal at ``spec["sampling_rate"]``.
        """
        x = resample_poly(x, spec["sampling_rate"], source_rate)
        threshold = spec["clip_iqr"] * np.subtract(*np.percentile(x, [75, 25]))
        x = np.clip(x, -threshold, threshold)
        return RobustScaler().fit_transform(x[:, None])[:, 0].astype(np.float32)
    return transform


def native_night(cfg: Config, spec: dict, subject: str, channels: list[str],
                 cache_dir: Path) -> np.ndarray:
    """A preprocessed night, cropped to the hypnogram rows our pipeline keeps.

    The cache file name hashes every setting that changes its content.
    The signal is cut to start at the first row's onset and to hold exactly
    one epoch per row, so U-Sleep's k-th output is the k-th row.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    spec : dict
        Native configuration.
    subject : str
        Subject id.
    channels : list of str
        Channels to read, in order.
    cache_dir : Path
        Cache folder of this configuration.

    Returns
    -------
    np.ndarray
        ``[n_channels, n_rows * epoch_samples]`` float32.

    Raises
    ------
    ValueError
        If the rows are not contiguous epochs or the recording is too short.
    """
    edf_path = subject_files(cfg)[subject][0]
    rows = night_epochs(cfg, subject)
    epoch_samples = spec["sampling_rate"] * spec["epoch_seconds"]
    if not np.allclose(np.diff(rows["onset"]), spec["epoch_seconds"]):
        raise ValueError(f"{subject}: hypnogram rows are not contiguous {spec['epoch_seconds']} s")
    key = json.dumps([channels, spec["sampling_rate"], spec["clip_iqr"], spec["epoch_seconds"],
                      float(rows["onset"][0]), len(rows["onset"]), str(edf_path)])
    path = cache_dir / f"{subject}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npy"

    def build() -> np.ndarray:
        """Read, preprocess and crop the night.

        Returns
        -------
        np.ndarray
            ``[n_channels, n_rows * epoch_samples]`` float32.

        Raises
        ------
        ValueError
            If the recording ends before the last row.
        """
        info = mne.io.read_raw_edf(edf_path, preload=False, verbose=False).info
        signal = read_channels(edf_path, channels, native_transform(spec, int(info["sfreq"])))
        start = int(round(rows["onset"][0] * spec["sampling_rate"]))
        stop = start + len(rows["onset"]) * epoch_samples
        if signal.shape[1] < stop:
            raise ValueError(f"{subject}: {signal.shape[1]} samples, rows need {stop}")
        return np.ascontiguousarray(signal[:, start:stop])
    return cached_night(path, build)


def predict_night(network: torch.nn.Module, night: np.ndarray, pairs: list[tuple[int, int]],
                  batch_pairs: int, device: torch.device) -> np.ndarray:
    """U-Sleep's softmax for every pair over the whole night.

    Parameters
    ----------
    network : torch.nn.Module
        AnySleep ``USleep`` (input ``[B, time, 2]``, output logits ``[B, epochs, 5]``).
    night : np.ndarray
        ``[n_channels, samples]`` preprocessed night.
    pairs : list of tuple
        ``(eeg_row, eog_row)`` indices into ``night``.
    batch_pairs : int
        Pairs per forward pass.
    device : torch.device
        Where to run.

    Returns
    -------
    np.ndarray
        ``[n_pairs, n_epochs, 5]`` softmax.
    """
    outputs = []
    with torch.inference_mode():
        for first in range(0, len(pairs), batch_pairs):
            chunk = pairs[first:first + batch_pairs]
            x = np.stack([night[[eeg, eog]].T for eeg, eog in chunk])
            logits = network(torch.from_numpy(x).to(device))
            outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(outputs)


def hard_vote(softmax: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """AnySleep's hard majority vote: most frequent argmax over pairs, ties broken at random.

    Parameters
    ----------
    softmax : np.ndarray
        ``[n_pairs, n_epochs, n_stages]``.
    rng : np.random.RandomState
        Source of the tie-breaking noise.

    Returns
    -------
    np.ndarray
        ``[n_epochs, n_stages]`` vote counts plus noise in ``[0, 1)``.
    """
    n_stages = softmax.shape[-1]
    counts = np.eye(n_stages)[softmax.argmax(-1)].sum(0)
    return counts + rng.random_sample(counts.shape)


def summarise(results: dict[str, dict]) -> dict:
    """Mean and standard deviation over checkpoints of every test metric.

    Parameters
    ----------
    results : dict
        ``checkpoint -> score_run result``.

    Returns
    -------
    dict
        ``metric -> {"mean", "sd", "values"}``.
    """
    metrics = next(iter(results.values()))["test"]
    summary = {}
    for metric in metrics:
        values = [result["test"][metric] for result in results.values()]
        summary[metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))
                           if len(values) > 1 else 0.0, "values": values}
    return summary


def run_native(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Zero-shot U-Sleep in its native configuration on the fold's test nights.

    Every night is preprocessed as U-Sleep expects (see
    :func:`native_transform`) and given whole to each released checkpoint
    once per (EEG, EOG) pair. The pairs are combined by the official rule,
    the sum of the softmax outputs (``utime/bin/predict.py``), and, as a
    secondary result, by AnySleep's hard majority vote. Each checkpoint and
    rule is scored with :func:`baselines.common.scoring.score_run` in its own
    folder; ``metrics.json`` holds them all with the mean and standard
    deviation over checkpoints.

    Parameters
    ----------
    spec : dict
        Native configuration.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test: ``settings["quick"]["test_subjects"]`` nights,
        ``spec["quick"]["pairs"]`` pairs, ``spec["quick"]["checkpoints"]`` checkpoints.

    Returns
    -------
    dict
        ``n_epochs``, ``test`` (mean over checkpoints, official rule),
        ``test_sd``, ``runs``, ``summary`` and ``parameters``.
    """
    cfg = load_reference_config(Path(settings["reference_run"]))
    subjects = load_fold(Path(settings["fold"]))["test"]
    checkpoints = [Path(path) for path in spec["checkpoints"]]
    if quick:
        subjects = subjects[:settings["quick"]["test_subjects"]]
        checkpoints = checkpoints[:spec["quick"]["checkpoints"]]
    pairs = eeg_eog_pairs(cfg, spec, quick)
    channels = list(dict.fromkeys(name for pair in pairs for name in pair))
    rows = [(channels.index(eeg), channels.index(eog)) for eeg, eog in pairs]
    cache_dir = Path(settings["cache_dir"]) / "usleep" / out_dir.name
    table = scored_epochs(cfg, subjects)

    timings = {}
    nights = {}
    for subject in subjects:
        start = time.perf_counter()
        nights[subject] = native_night(cfg, spec, subject, channels, cache_dir)
        timings[f"preprocess_{subject}_s"] = time.perf_counter() - start

    runs, parameters = {}, None
    for checkpoint in checkpoints:
        network = pretrained_usleep(checkpoint, **spec["architecture"]).to(cfg.device)
        parameters = sum(p.numel() for p in network.parameters())
        rng = np.random.RandomState(spec["tie_seed"])
        columns = {rule: {"subject": [], "onset": [], "probability": []}
                   for rule in ("soft", "hard")}
        for subject in subjects:
            start = time.perf_counter()
            softmax = predict_night(network, nights[subject], rows, spec["batch_pairs"],
                                    cfg.device)
            timings[f"predict_{checkpoint.stem}_{subject}_s"] = time.perf_counter() - start
            onsets = night_epochs(cfg, subject)["onset"]
            if softmax.shape[1] != len(onsets):
                raise ValueError(f"{subject}: {softmax.shape[1]} outputs for {len(onsets)} rows")
            scored = np.isin(np.round(onsets, 3), np.round(table[subject]["onset"], 3))
            votes = {"soft": softmax.sum(0), "hard": hard_vote(softmax, rng)}
            for rule, vote in votes.items():
                columns[rule]["subject"] += [subject] * int(scored.sum())
                columns[rule]["onset"].append(onsets[scored])
                columns[rule]["probability"].append(vote[scored] / vote[scored].sum(-1,
                                                                                    keepdims=True))
        for rule, column in columns.items():
            probability = np.concatenate(column["probability"])
            path = out_dir / f"{checkpoint.stem}_{rule}" / "predictions.npz"
            save_predictions(path, np.array(column["subject"]), np.concatenate(column["onset"]),
                             probability.argmax(-1), probability)
            runs[f"{checkpoint.stem}_{rule}"] = score_run(cfg, table, path)
        del network
        torch.cuda.empty_cache()

    summary = {rule: summarise({name: result for name, result in runs.items()
                                if name.endswith(f"_{rule}")}) for rule in ("soft", "hard")}
    result = {"n_epochs": next(iter(runs.values()))["n_epochs"],
              "test": {metric: value["mean"] for metric, value in summary["soft"].items()},
              "test_sd": {metric: value["sd"] for metric, value in summary["soft"].items()},
              "rule": "sum of softmax over pairs (official); hard vote in summary['hard']",
              "summary": summary, "runs": runs, "parameters": parameters,
              "n_pairs": len(pairs), "checkpoints": [str(path) for path in checkpoints],
              "subjects": subjects, "timings": timings, "quick": quick}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result
