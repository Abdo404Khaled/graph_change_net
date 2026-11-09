from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data
from baselines.cbramod.model import SequenceModel, nearest_neighbour_chain
from baselines.common.native import cached_night, night_epochs, read_channels, subject_files
from baselines.common.ours import run_ours
from baselines.common.reference import load_fold, load_reference_config, scored_epochs
from baselines.common.scoring import save_predictions, score_run
from config import Config
from evaluate import scores

ROOT = Path(__file__).resolve().parents[2]


def run(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """Run CBraMod under its native ISRUC recipe or under our configuration.

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml`` or ``ours.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder of this run.
    quick : bool
        Smoke test on a few nights and steps.

    Returns
    -------
    dict
        The scores, parameter count and resource use.

    Raises
    ------
    ValueError
        If ``spec["protocol"]`` is neither ``native`` nor ``ours``.
    """
    if spec["protocol"] == "native":
        return run_native(spec, settings, out_dir, quick)
    if spec["protocol"] == "ours":
        return run_our_configuration(spec, settings, out_dir, quick)
    raise ValueError(f"unknown protocol {spec['protocol']!r}")


def seed_everything(seed: int) -> None:
    """Seed every generator as upstream's ``setup_seed`` does.

    Parameters
    ----------
    seed : int
        Seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_network(spec: dict, channel_order: np.ndarray, n_patches: int,
                  n_classes: int) -> SequenceModel:
    """CBraMod with its ISRUC head, sized for the configuration.

    Parameters
    ----------
    spec : dict
        Configuration (``weights`` and ``model``).
    channel_order : np.ndarray
        Input electrode index at each backbone position.
    n_patches : int
        Patches per epoch.
    n_classes : int
        Number of stages.

    Returns
    -------
    SequenceModel
        The network on the CPU.
    """
    model = spec["model"]
    return SequenceModel(ROOT / spec["weights"], model["backbone"], channel_order, n_patches,
                         model["patch_samples"], model["head_hidden"], model["sequence_heads"],
                         model["sequence_feedforward"], model["sequence_layers"], n_classes)


def run_our_configuration(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """CBraMod's architecture on our data protocol and our stage-2 trainer.

    The 83 montage electrodes at 1 kHz are cut into 1 s patches of 1000
    points (30 per epoch) and permuted into the order ``model.channel_order``
    names; the pretrained backbone, the stretched patch encoder, the flatten
    head and the bidirectional sequence transformer are trained by
    :func:`baselines.common.ours.run_ours` with ``spec["overrides"]``.

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
        Scores from :func:`run_ours`, with resource use.

    Raises
    ------
    ValueError
        If an epoch is not a whole number of patches, or the channel order is unknown.
    """
    cfg = load_reference_config(ROOT / settings["reference_run"], spec["overrides"])
    channels, coords = data.read_montage(data.find_montage(cfg.paths.data_root))
    epoch_samples = int(round(cfg.data.window_s * cfg.data.sfreq))
    patch_samples = spec["model"]["patch_samples"]
    if epoch_samples % patch_samples:
        raise ValueError(f"an epoch of {epoch_samples} samples is not whole patches")
    orders = {"montage": np.arange(len(channels)),
              "nearest_neighbour_chain": nearest_neighbour_chain(coords)}
    if spec["model"]["channel_order"] not in orders:
        raise ValueError(f"unknown channel order {spec['model']['channel_order']!r}")
    order = orders[spec["model"]["channel_order"]]
    torch.manual_seed(cfg.seed)
    network = build_network(spec, order, epoch_samples // patch_samples, len(cfg.data.stages))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = run_ours(network, settings, spec["overrides"], out_dir, quick)
    result["resources"] = {"seconds": time.time() - start, **peak_memory()}
    result["channel_order"] = [channels[index] for index in order]
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def peak_memory() -> dict:
    """Largest GPU memory held by tensors so far.

    Returns
    -------
    dict
        ``max_memory_allocated_gib`` (0 without CUDA).
    """
    if not torch.cuda.is_available():
        return {"max_memory_allocated_gib": 0.0}
    return {"max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def preprocess_night(edf_path: Path, spec: dict, recorded_hz: float) -> np.ndarray:
    """One night as the ISRUC script prepares it, from ANPHY's 1 kHz recording.

    Each derivation (``[a, b]`` read as ``a - b``, in microvolts) is
    resampled to ``sampling_hz`` with ``scipy.signal.resample_poly`` while it
    is read; the six signals are then band-passed with MNE's FIR filter
    (``fir_design='firwin'``, as ``raw.filter`` in
    ``prepare_ISRUC_1.py``) and notch-filtered at the mains frequency.

    Parameters
    ----------
    edf_path : Path
        ANPHY recording.
    spec : dict
        Native configuration (``data``).
    recorded_hz : float
        Sampling rate of the recording.

    Returns
    -------
    np.ndarray
        ``[n_derivations, samples]`` float32 microvolts at ``sampling_hz``.
    """
    setting = spec["data"]
    up, down = int(setting["sampling_hz"]), int(recorded_hz)

    def resample(signal: np.ndarray) -> np.ndarray:
        """Polyphase resampling of one derivation.

        Parameters
        ----------
        signal : np.ndarray
            1-D microvolts at the recording rate.

        Returns
        -------
        np.ndarray
            1-D microvolts at ``sampling_hz``.
        """
        return resample_poly(signal, up, down)

    signal = read_channels(edf_path, setting["derivations"], resample).astype(np.float64)
    signal = mne.filter.filter_data(signal, setting["sampling_hz"], setting["highpass"],
                                    setting["lowpass"], fir_design="firwin", verbose=False)
    signal = mne.filter.notch_filter(signal, setting["sampling_hz"], [setting["notch"]],
                                     verbose=False)
    return signal.astype(np.float32)


class NightSequences(Dataset):
    """Runs of consecutive 30 s epochs of whole nights, as ISRUC's sequences.

    Each night is cut from its first hypnogram row into non-overlapping runs
    of ``sequence_epochs`` epochs; when the night is not a multiple of that,
    one more run ends at its last epoch (ISRUC drops those epochs instead).
    Every row is an epoch, labelled with its stage or -1 when unscored.
    Signals are divided by ``unit_uv`` (units of 100 uV).

    Parameters
    ----------
    nights : dict
        ``subject -> (signal [C, samples], starts [n_epochs], stage [n_epochs], onset [n_epochs])``.
    subjects : list of str
        Nights to include, in this order.
    sequence_epochs : int
        Epochs per run.
    epoch_samples : int
        Samples per epoch.
    unit_uv : float
        Divisor applied to the microvolt signal.

    Raises
    ------
    ValueError
        If a night is shorter than one run.
    """

    def __init__(self, nights: dict, subjects: list[str], sequence_epochs: int,
                 epoch_samples: int, unit_uv: float):
        self.nights, self.subjects = nights, subjects
        self.sequence_epochs, self.epoch_samples, self.unit_uv = (sequence_epochs, epoch_samples,
                                                                  unit_uv)
        self.items = []
        for index, subject in enumerate(subjects):
            n_epochs = len(nights[subject][1])
            if n_epochs < sequence_epochs:
                raise ValueError(f"{subject} has fewer than {sequence_epochs} epochs")
            first = list(range(0, n_epochs - sequence_epochs + 1, sequence_epochs))
            if n_epochs % sequence_epochs:
                first.append(n_epochs - sequence_epochs)
            self.items.extend((index, start) for start in first)

    def __len__(self) -> int:
        """Number of runs.

        Returns
        -------
        int
            Runs over all nights.
        """
        return len(self.items)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One run and its labels.

        Parameters
        ----------
        item : int
            Run index.

        Returns
        -------
        x : torch.Tensor
            ``[sequence_epochs, C, epoch_samples]`` float32.
        y : torch.Tensor
            ``[sequence_epochs]`` stage indices, -1 where unscored.
        """
        index, start = self.items[item]
        signal, starts, stage, _ = self.nights[self.subjects[index]]
        rows = range(start, start + self.sequence_epochs)
        x = np.stack([signal[:, starts[row]:starts[row] + self.epoch_samples] for row in rows])
        return (torch.from_numpy(x.astype(np.float32) / self.unit_uv),
                torch.from_numpy(stage[start:start + self.sequence_epochs].astype(np.int64)))


def load_nights(cfg: Config, spec: dict, subjects: list[str], cache_dir: Path) -> dict:
    """Preprocessed nights and the epoch grid of our pipeline.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    spec : dict
        Native configuration.
    subjects : list of str
        Nights to load.
    cache_dir : Path
        Folder of the preprocessed nights.

    Returns
    -------
    dict
        ``subject -> (signal, starts, stage, onset)``; ``signal`` is memory-mapped.

    Raises
    ------
    ValueError
        If an epoch runs past the end of the resampled night.
    """
    files = subject_files(cfg)
    sampling_hz = spec["data"]["sampling_hz"]
    epoch_samples = int(round(spec["data"]["epoch_s"] * sampling_hz))
    nights = {}
    for subject in tqdm(subjects, desc="native nights"):
        edf_path = files[subject][0]
        signal = cached_night(cache_dir / f"{subject}.npy",
                              lambda: preprocess_night(edf_path, spec, cfg.data.sfreq))
        epochs = night_epochs(cfg, subject)
        starts = np.round(epochs["onset"] * sampling_hz).astype(np.int64)
        if starts.max() + epoch_samples > signal.shape[1]:
            raise ValueError(f"{subject}: an epoch runs past the end of the night")
        nights[subject] = (signal, starts, epochs["stage"].astype(np.int64), epochs["onset"])
    return nights


@torch.no_grad()
def predict_nights(model: nn.Module, dataset: NightSequences, batch_size: int,
                   device: torch.device) -> dict:
    """One prediction per epoch, from the first run that holds it.

    Runs are read in order, so an epoch covered twice (by a non-overlapping
    run and by the night's last, right-aligned run) keeps the
    non-overlapping run's prediction.

    Parameters
    ----------
    model : nn.Module
        Network.
    dataset : NightSequences
        Runs of the nights to predict.
    batch_size : int
        Runs per forward pass.
    device : torch.device
        Device.

    Returns
    -------
    dict
        ``subject -> probability [n_epochs, n_classes]``.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False)
    probabilities = [torch.softmax(model(x.to(device)), -1).cpu() for x, _ in loader]
    probability = torch.cat(probabilities).numpy()
    out = {}
    for (index, start), runs in zip(dataset.items, probability):
        subject = dataset.subjects[index]
        if subject not in out:
            out[subject] = np.full((len(dataset.nights[subject][1]), runs.shape[-1]), np.nan)
        target = out[subject][start:start + len(runs)]
        empty = np.isnan(target[:, 0])
        target[empty] = runs[empty]
    return out


def native_scores(cfg: Config, nights: dict, probability: dict) -> dict:
    """Upstream's metrics and ours on the scored epochs of some nights.

    Parameters
    ----------
    cfg : Config
        Reference configuration (stage names).
    nights : dict
        Output of :func:`load_nights`.
    probability : dict
        Output of :func:`predict_nights`.

    Returns
    -------
    dict
        Our :func:`evaluate.scores`, plus upstream's ``balanced_accuracy`` and
        ``weighted_f1`` (pooled, as ``finetune_evaluator.py``).
    """
    true = np.concatenate([nights[s][2] for s in probability])
    predicted = np.concatenate([probability[s].argmax(-1) for s in probability])
    scored = true >= 0
    true, predicted = true[scored], predicted[scored]
    return {**scores(cfg, true, predicted),
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "weighted_f1": float(f1_score(true, predicted, average="weighted")),
            "upstream_kappa": float(cohen_kappa_score(true, predicted))}


def run_native(spec: dict, settings: dict, out_dir: Path, quick: bool) -> dict:
    """CBraMod's ISRUC fine-tuning recipe on ANPHY, scored on the fold's test nights.

    Six contralateral-mastoid derivations at 200 Hz in units of 100 uV, runs
    of 20 epochs, the pretrained backbone fully fine-tuned with upstream's
    optimiser (AdamW, backbone and head learning rates), per-step cosine
    schedule, gradient clipping and label smoothing; the model with the best
    validation kappa is tested. Unscored positions (-1) are left out of the
    loss. A batch of ``train.batch_size`` runs is processed in chunks of
    ``train.micro_batch_size`` with the gradients summed, which gives the
    same update (the network has no batch statistics).

    Parameters
    ----------
    spec : dict
        Contents of ``native.yaml``.
    settings : dict
        Contents of ``baselines/settings.yaml``.
    out_dir : Path
        Output folder.
    quick : bool
        Smoke test: ``settings["quick"]`` subjects per split, ``epochs``
        epochs of at most ``native_steps`` steps.

    Returns
    -------
    dict
        Scores of :func:`score_run`, upstream's metrics, history, parameter
        count and resource use.
    """
    quick_settings = settings["quick"]
    cfg = load_reference_config(ROOT / settings["reference_run"])
    split = load_fold(ROOT / settings["fold"])
    if quick:
        split = {name: split[name][:quick_settings[f"{name}_subjects"]] for name in split}
    cache_dir = ROOT / settings["cache_dir"] / "cbramod" / ("native_quick" if quick else "native")
    nights = load_nights(cfg, spec, sum(split.values(), []), cache_dir)
    setting, train = spec["data"], spec["train"]
    epoch_samples = int(round(setting["epoch_s"] * setting["sampling_hz"]))
    datasets = {name: NightSequences(nights, split[name], setting["sequence_epochs"],
                                     epoch_samples, setting["unit_uv"]) for name in split}

    seed_everything(spec["seed"])
    device = cfg.device
    n_derivations = len(setting["derivations"])
    model = build_network(spec, np.arange(n_derivations),
                          epoch_samples // spec["model"]["patch_samples"],
                          len(cfg.data.stages)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    backbone = [p for name, p in model.named_parameters() if "backbone" in name]
    others = [p for name, p in model.named_parameters() if "backbone" not in name]
    optimiser = torch.optim.AdamW([{"params": backbone, "lr": train["lr"]},
                                   {"params": others, "lr": train["head_lr"]}],
                                  weight_decay=train["weight_decay"])
    loader = DataLoader(datasets["train"], train["batch_size"], shuffle=True,
                        num_workers=train["num_workers"])
    epochs = quick_settings["epochs"] if quick else train["epochs"]
    steps = min(len(loader), quick_settings["native_steps"]) if quick else len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * steps,
                                                           eta_min=train["min_lr"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    best_kappa, best_state, history, step_seconds = -float("inf"), None, [], []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_start, losses = time.time(), []
        for step, (x, y) in enumerate(tqdm(loader, total=steps, leave=False,
                                           desc=f"epoch {epoch + 1}")):
            if step >= steps:
                break
            step_start = time.time()
            losses.append(accumulated_step(model, optimiser, x, y, train, device))
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_seconds.append(time.time() - step_start)
        validation = native_scores(cfg, nights, predict_nights(
            model, datasets["val"], train["eval_batch_size"], device))
        validation.update({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                           "seconds": time.time() - epoch_start})
        history.append(validation)
        print(f"epoch {epoch + 1}: loss {validation['train_loss']:.4f} "
              f"val kappa {validation['upstream_kappa']:.4f} "
              f"bacc {validation['balanced_accuracy']:.4f} wf1 {validation['weighted_f1']:.4f}")
        if validation["upstream_kappa"] > best_kappa:
            best_kappa = validation["upstream_kappa"]
            best_state = copy.deepcopy({key: value.cpu() for key, value in
                                        model.state_dict().items()})
    model.load_state_dict(best_state)

    probability = predict_nights(model, datasets["test"], train["eval_batch_size"], device)
    names, onsets, predicted, probabilities = [], [], [], []
    for subject, values in probability.items():
        stage, onset = nights[subject][2], nights[subject][3]
        scored = stage >= 0
        names.extend([subject] * int(scored.sum()))
        onsets.extend(onset[scored].tolist())
        predicted.append(values[scored].argmax(-1))
        probabilities.append(values[scored])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.npz"
    save_predictions(path, np.array(names), np.array(onsets), np.concatenate(predicted),
                     np.concatenate(probabilities))
    result = score_run(cfg, scored_epochs(cfg, split["test"]), path)
    result.update({"test_native_metrics": native_scores(cfg, nights, probability),
                   "parameters": parameters, "val_best_kappa": best_kappa, "quick": quick,
                   "history": history,
                   "resources": {"seconds": time.time() - started,
                                 "seconds_per_step": float(np.mean(step_seconds)),
                                 "steps_per_epoch": steps, **peak_memory()}})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def accumulated_step(model: nn.Module, optimiser: torch.optim.Optimizer, x: torch.Tensor,
                     y: torch.Tensor, train: dict, device: torch.device) -> float:
    """One optimiser step on a batch, in chunks whose gradients add up to the batch's.

    The loss is upstream's label-smoothed cross entropy over every labelled
    position of the batch (``ignore_index=-1``), i.e. the summed loss of the
    chunks divided by the batch's labelled positions.

    Parameters
    ----------
    model : nn.Module
        Network in training mode.
    optimiser : torch.optim.Optimizer
        Optimiser.
    x : torch.Tensor
        Runs ``[B, S, C, samples]``.
    y : torch.Tensor
        Labels ``[B, S]``.
    train : dict
        Training settings (``micro_batch_size``, ``label_smoothing``, ``clip_value``).
    device : torch.device
        Device.

    Returns
    -------
    float
        The batch's mean loss.
    """
    optimiser.zero_grad()
    labelled = max(int((y >= 0).sum()), 1)
    total = 0.0
    for chunk_x, chunk_y in zip(x.split(train["micro_batch_size"]),
                                y.split(train["micro_batch_size"])):
        chunk_y = chunk_y.to(device)
        logits = model(chunk_x.to(device))
        loss = F.cross_entropy(logits.transpose(1, 2), chunk_y, ignore_index=-1,
                               label_smoothing=train["label_smoothing"], reduction="sum")
        (loss / labelled).backward()
        total += loss.item()
    nn.utils.clip_grad_norm_(model.parameters(), train["clip_value"])
    optimiser.step()
    return total / labelled
