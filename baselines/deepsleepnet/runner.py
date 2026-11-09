"""Run DeepSleepNet on fold 0 in its own configuration (``native``) or in ours (``ours``)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

import data
from baselines.common.native import cached_night, night_epochs, read_channels, subject_files
from baselines.common.ours import run_ours
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from baselines.deepsleepnet.model import (DeepSleepNet, PretrainNet, State, build_features,
                                          build_network)
from config import Config
from evaluate import scores


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Train and score DeepSleepNet under one configuration.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        Scores, parameter count, timings and peak GPU memory.

    Raises
    ------
    ValueError
        If the configuration is neither ``native`` nor ``ours``.
    """
    torch.manual_seed(spec["seed"])
    if spec["configuration"] == "native":
        return run_native(spec, settings, out_dir, quick)
    if spec["configuration"] == "ours":
        return run_our_configuration(spec, settings, out_dir, quick)
    raise ValueError(f"unknown configuration {spec['configuration']!r}")


def peak_memory_gib() -> float:
    """Peak allocated GPU memory since the last reset, in GiB (0 without CUDA).

    Returns
    -------
    float
        Peak memory.
    """
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


def reset_peak_memory() -> None:
    """Reset the peak GPU memory counter when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def synchronise() -> None:
    """Wait for queued GPU work, so wall-clock timings are complete."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_our_configuration(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """DeepSleepNet under our data protocol and our stage-2 trainer, from scratch.

    Parameters
    ----------
    spec : dict
        Contents of ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test.

    Returns
    -------
    dict
        Scores from :func:`baselines.common.ours.run_ours`, with timing and memory.

    Raises
    ------
    ValueError
        If the sampling rate or channel count differs from the reference data.
    """
    cfg = load_reference_config(Path(settings["reference_run"]))
    channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
    signal = spec["signal"]
    if signal["sampling_rate"] != cfg.data.sfreq or signal["in_channels"] != len(channels):
        raise ValueError(f"ours.yaml expects {signal['in_channels']} channels at "
                         f"{signal['sampling_rate']} Hz; the reference data have "
                         f"{len(channels)} at {cfg.data.sfreq} Hz")
    n_samples = int(round(cfg.data.window_s * cfg.data.sfreq))
    features = build_features(spec["model"], signal["in_channels"], n_samples,
                              signal["sampling_rate"])
    network = build_network(spec["model"], features, len(cfg.data.stages))
    reset_peak_memory()
    start = time.perf_counter()
    result = run_ours(network, settings, spec["overrides"], out_dir, quick)
    result.update({"seconds": time.perf_counter() - start, "peak_memory_gib": peak_memory_gib(),
                   "feature_size": features.feature_size,
                   "first_kernels": [features.small.first.kernel, features.large.first.kernel],
                   "first_strides": [features.small.first.stride, features.large.first.stride]})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def load_night(cfg: Config, spec: dict, subject: str, files: dict, cache_dir: Path
               ) -> dict[str, np.ndarray]:
    """Scored epochs of one night in the native form: FPZ - CZ, resampled, unfiltered, in µV.

    The bipolar signal of the whole night is resampled once with
    :func:`scipy.signal.resample_poly` and cached; the epochs are cut at our
    pipeline's hypnogram onsets, and rows without a stage (lights on) are
    left out, as upstream cuts out its UNKNOWN and MOVEMENT rows.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    spec : dict
        Contents of ``native.yaml``.
    subject : str
        Subject id.
    files : dict
        Output of :func:`baselines.common.native.subject_files`.
    cache_dir : Path
        Folder of the resampled nights.

    Returns
    -------
    dict
        ``x`` ``[N, 1, epoch samples]`` float32, ``stage`` and ``onset`` of the
        scored epochs, in time order.

    Raises
    ------
    ValueError
        If an epoch runs past the end of the resampled night.
    """
    signal_spec = spec["signal"]

    def resample(x: np.ndarray) -> np.ndarray:
        """Resample one channel to the native rate.

        Parameters
        ----------
        x : np.ndarray
            1-D signal at the recording rate.

        Returns
        -------
        np.ndarray
            1-D signal at ``signal.sampling_rate``.
        """
        return resample_poly(x, signal_spec["resample_up"], signal_spec["resample_down"])

    rate = signal_spec["sampling_rate"]
    name = f"{subject}_{'-'.join(signal_spec['channel'])}_{rate:g}hz.npy"
    night = cached_night(cache_dir / name, lambda: read_channels(
        files[subject][0], [signal_spec["channel"]], resample))
    epochs = night_epochs(cfg, subject)
    scored = epochs["stage"] >= 0
    n_samples = int(round(signal_spec["epoch_s"] * rate))
    starts = np.round(epochs["onset"][scored] * rate).astype(np.int64)
    if starts.max() + n_samples > night.shape[1]:
        raise ValueError(f"{subject}: an epoch runs past the resampled night")
    x = np.stack([night[:, start:start + n_samples] for start in starts]).astype(np.float32)
    return {"x": x, "stage": epochs["stage"][scored].astype(np.int64),
            "onset": epochs["onset"][scored].astype(np.float64)}


def trim_wake(stage: np.ndarray, edge: int) -> np.ndarray:
    """Upstream's wake trimming (``prepare_physionet.py:187-198``).

    Parameters
    ----------
    stage : np.ndarray
        Stage index per epoch (0 is W).
    edge : int
        Epochs of wake kept before the first and after the last non-W epoch.

    Returns
    -------
    np.ndarray
        Indices of the kept epochs.
    """
    sleep = np.flatnonzero(stage != 0)
    first, last = max(sleep[0] - edge, 0), min(sleep[-1] + edge, len(stage) - 1)
    return np.arange(first, last + 1)


def oversample(stage: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Upstream's class balancing by oversampling (``utils.py:34-65``), as indices.

    Every class is repeated ``floor(largest / size)`` times and topped up with
    a random subset of itself to the size of the largest class.

    Parameters
    ----------
    stage : np.ndarray
        Stage index per training epoch.
    generator : np.random.Generator
        Random draws.

    Returns
    -------
    np.ndarray
        Indices into the training epochs, every class equally often.
    """
    classes = np.unique(stage)
    largest = max(int((stage == label).sum()) for label in classes)
    chosen = []
    for label in classes:
        members = np.flatnonzero(stage == label)
        repeated = np.repeat(members, largest // len(members))
        chosen.extend([repeated, generator.permutation(members)[:largest - len(repeated)]])
    return np.concatenate(chosen)


def first_conv_decay(network: nn.Module, spec: dict) -> torch.Tensor:
    """Upstream's L2 term, ``wd * l2_loss(W) = wd * sum(W ** 2) / 2``, on the first convolutions.

    The code sets ``wd=1e-3`` on the small-filter branch's first convolution
    only (``model.py:75``) and none on the large one (``model.py:116``); the
    paper says both. ``weight_decay`` in ``native.yaml`` follows the code.

    Parameters
    ----------
    network : nn.Module
        A model with a ``features`` DeepFeatureNet.
    spec : dict
        ``weight_decay`` section.

    Returns
    -------
    torch.Tensor
        The penalty.
    """
    features = network.features
    return (spec["small_first"] * features.small.first.convolution.weight.pow(2).sum() / 2
            + spec["large_first"] * features.large.first.convolution.weight.pow(2).sum() / 2)


def pretrain(spec: dict, nights: dict, model: PretrainNet, device: torch.device,
             generator: np.random.Generator, max_steps: int | None) -> list[dict]:
    """Step 1: the CNNs with their own softmax on the oversampled training epochs.

    Adam, mean cross-entropy plus the first-convolution L2, shuffled batches
    of single epochs, the last incomplete batch dropped, no clipping
    (``trainer.py:216-442``).

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``.
    nights : dict
        Training nights from :func:`load_night`, wake-trimmed.
    model : PretrainNet
        Model trained in place.
    device : torch.device
        Device.
    generator : np.random.Generator
        Random draws (oversampling and shuffling).
    max_steps : int or None
        Steps per epoch in smoke tests; None runs every batch.

    Returns
    -------
    list of dict
        Mean loss, accuracy, seconds per step and peak memory per epoch.
    """
    settings = spec["pretrain"]
    x = np.concatenate([night["x"] for night in nights.values()])
    stage = np.concatenate([night["stage"] for night in nights.values()])
    balanced = oversample(stage, generator)
    optimiser = torch.optim.Adam(model.parameters(), settings["lr"],
                                 betas=tuple(settings["betas"]), eps=settings["eps"])
    history = []
    for _ in range(settings["epochs"]):
        model.train()
        reset_peak_memory()
        order = balanced[generator.permutation(len(balanced))]
        total, correct, seen, steps, start = 0.0, 0, 0, 0, time.perf_counter()
        for begin in range(0, len(order) - settings["batch_size"] + 1, settings["batch_size"]):
            batch = order[begin:begin + settings["batch_size"]]
            inputs = torch.from_numpy(x[batch]).to(device)
            target = torch.from_numpy(stage[batch]).to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, target) + first_conv_decay(model, spec["weight_decay"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            correct += int((logits.argmax(1) == target).sum())
            seen += len(batch)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        synchronise()
        history.append({"loss": total / max(steps, 1), "accuracy": correct / max(seen, 1),
                        "steps": steps, "balanced_epochs": int(len(balanced)),
                        "seconds_per_step": (time.perf_counter() - start) / max(steps, 1),
                        "peak_memory_gib": peak_memory_gib()})
    return history


def detach(state: State) -> State:
    """Cut a carried LSTM state from the graph (truncated backpropagation through time).

    Parameters
    ----------
    state : State
        State of both directions.

    Returns
    -------
    State
        The same values without history.
    """
    return tuple([(hidden.detach(), cell.detach()) for hidden, cell in direction]
                 for direction in state)


def finetune(spec: dict, nights: dict, network: DeepSleepNet, device: torch.device,
             max_steps: int | None) -> list[dict]:
    """Step 2: the whole network on sequential, non-oversampled nights.

    Each night, in the fold's order, is split into ``batch_size`` contiguous
    sub-sequences that are read ``seq_length`` epochs at a time; the LSTM
    state is carried from one step to the next and reset at each night
    (``trainer.py:480-541``, ``utils.py:105-128``). Two Adam learning rates
    (pretrained CNNs and the rest), global-norm clipping over all parameters,
    and the sequence loss summed over epochs and averaged over sequences, plus
    the first-convolution L2 (``model.py:400-422``, ``optimize.py:33-71``).

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``.
    nights : dict
        Training nights from :func:`load_night`, wake-trimmed.
    network : DeepSleepNet
        Network trained in place, its CNNs already pretrained.
    device : torch.device
        Device.
    max_steps : int or None
        Steps per epoch in smoke tests; None runs every step.

    Returns
    -------
    list of dict
        Mean loss, seconds per step and peak memory per epoch.
    """
    settings = spec["finetune"]
    cnn = list(network.features.parameters())
    cnn_ids = {id(parameter) for parameter in cnn}
    rest = [parameter for parameter in network.parameters() if id(parameter) not in cnn_ids]
    optimiser = torch.optim.Adam([{"params": cnn, "lr": settings["cnn_lr"]},
                                  {"params": rest, "lr": settings["sequence_lr"]}],
                                 betas=tuple(settings["betas"]), eps=settings["eps"])
    batch, length = settings["batch_size"], settings["seq_length"]
    history = []
    for _ in range(settings["epochs"]):
        network.train()
        reset_peak_memory()
        total, steps, start = 0.0, 0, time.perf_counter()
        for night in nights.values():
            stride = len(night["stage"]) // batch
            state = network.zero_state(batch, device)
            for offset in range(0, (stride // length) * length, length):
                rows = np.arange(batch)[:, None] * stride + offset + np.arange(length)[None, :]
                inputs = torch.from_numpy(night["x"][rows]).to(device)
                target = torch.from_numpy(night["stage"][rows]).to(device)
                logits, state = network.step(inputs, state)
                state = detach(state)
                loss = (F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum")
                        / batch + first_conv_decay(network, spec["weight_decay"]))
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), settings["clip_norm"])
                optimiser.step()
                total += loss.item()
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            if max_steps is not None and steps >= max_steps:
                break
        synchronise()
        history.append({"loss": total / max(steps, 1), "steps": steps,
                        "seconds_per_step": (time.perf_counter() - start) / max(steps, 1),
                        "peak_memory_gib": peak_memory_gib()})
    return history


@torch.no_grad()
def predict_night(network: DeepSleepNet, night: dict, length: int,
                  device: torch.device) -> np.ndarray:
    """Stage probabilities of every epoch of a night, as ``predict.py`` scores it.

    One sequence, read ``length`` epochs at a time with the state of both
    directions carried over (``predict.py:473-541``); the last, shorter
    chunk is also read, so every epoch gets a prediction (upstream drops
    the last ``n mod length`` epochs).

    Parameters
    ----------
    network : DeepSleepNet
        Trained network.
    night : dict
        Night from :func:`load_night`.
    length : int
        Epochs per chunk.
    device : torch.device
        Device.

    Returns
    -------
    np.ndarray
        ``[n_epochs, n_classes]`` probabilities.
    """
    network.eval()
    state = network.zero_state(1, device)
    probabilities = []
    for begin in range(0, len(night["stage"]), length):
        inputs = torch.from_numpy(night["x"][begin:begin + length][None]).to(device)
        logits, state = network.step(inputs, state)
        probabilities.append(torch.softmax(logits[0], dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def run_native(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """DeepSleepNet in its own configuration on the fold, scored on every scored test epoch.

    Training nights are wake-trimmed; validation nights only give a logged
    sanity score of the final weights; there is no model selection.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test: ``quick.native_nights`` training nights, the quick
        numbers of validation and test nights, one epoch of each step with at
        most ``quick.native_steps`` steps.

    Returns
    -------
    dict
        Test scores, validation sanity scores, parameter count, histories,
        timings and peak memory.
    """
    cfg = load_reference_config(Path(settings["reference_run"]))
    device = cfg.device
    generator = np.random.default_rng(spec["seed"])
    split = load_fold(Path(settings["fold"]))
    quick_settings = settings["quick"]
    max_steps = None
    if quick:
        split = {"train": split["train"][:quick_settings["native_nights"]],
                 "val": split["val"][:quick_settings["val_subjects"]],
                 "test": split["test"][:quick_settings["test_subjects"]]}
        spec = {**spec, "pretrain": {**spec["pretrain"], "epochs": quick_settings["epochs"]},
                "finetune": {**spec["finetune"], "epochs": quick_settings["epochs"]}}
        max_steps = quick_settings["native_steps"]
    cache_dir = (Path(settings["cache_dir"]) / "deepsleepnet"
                 / (spec["configuration"] + ("_quick" if quick else "")))
    files = subject_files(cfg)
    start = time.perf_counter()
    nights = {name: {subject: load_night(cfg, spec, subject, files, cache_dir)
                     for subject in split[name]} for name in split}
    loading_seconds = time.perf_counter() - start
    for subject, night in nights["train"].items():
        kept = trim_wake(night["stage"], spec["signal"]["wake_edge_epochs"])
        nights["train"][subject] = {key: values[kept] for key, values in night.items()}

    n_samples = int(round(spec["signal"]["epoch_s"] * spec["signal"]["sampling_rate"]))
    n_classes = len(cfg.data.stages)
    features = build_features(spec["model"], 1, n_samples, spec["signal"]["sampling_rate"])
    pretrain_model = PretrainNet(features, n_classes).to(device)
    pretrain_history = pretrain(spec, nights["train"], pretrain_model, device, generator,
                                max_steps)
    network = build_network(spec["model"], pretrain_model.features, n_classes).to(device)
    finetune_history = finetune(spec, nights["train"], network, device, max_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), out_dir / "deepsleepnet_last.pt")

    length = spec["finetune"]["seq_length"]
    reset_peak_memory()
    start = time.perf_counter()
    val_true, val_predicted = [], []
    for night in nights["val"].values():
        val_true.append(night["stage"])
        val_predicted.append(predict_night(network, night, length, device).argmax(1))
    names, onsets, probabilities = [], [], []
    for subject, night in nights["test"].items():
        probabilities.append(predict_night(network, night, length, device))
        names.extend([subject] * len(night["stage"]))
        onsets.append(night["onset"])
    synchronise()
    n_predicted = sum(len(night["stage"]) for name in ("val", "test")
                      for night in nights[name].values())
    probability = np.concatenate(probabilities)
    path = out_dir / "predictions.npz"
    save_predictions(path, np.array(names), np.concatenate(onsets), probability.argmax(1),
                     probability)
    result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
    result.update({
        "parameters": sum(p.numel() for p in network.parameters()),
        "pretrain_head_parameters": sum(p.numel() for p in pretrain_model.classifier.parameters()),
        "val_sanity": scores(cfg, np.concatenate(val_true), np.concatenate(val_predicted)),
        "split": split, "quick": quick,
        "n_train_epochs": int(sum(len(night["stage"]) for night in nights["train"].values())),
        "feature_size": features.feature_size,
        "first_kernels": [features.small.first.kernel, features.large.first.kernel],
        "first_strides": [features.small.first.stride, features.large.first.stride],
        "loading_seconds": loading_seconds,
        "inference_ms_per_epoch": 1000 * (time.perf_counter() - start) / max(n_predicted, 1),
        "inference_peak_memory_gib": peak_memory_gib(),
        "pretrain_history": pretrain_history, "finetune_history": finetune_history})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result
