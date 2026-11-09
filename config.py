from __future__ import annotations

import dataclasses
import subprocess
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from components import check_model

CONFIG_DIR = Path("configs")
DEFAULTS = CONFIG_DIR / "default.yaml"
QUICK = CONFIG_DIR / "quick.yaml"
PLAN = Path("experiments.yaml")


@dataclass
class PathsConfig:
    """Where the data, the preprocessing cache and the run outputs live.

    Parameters
    ----------
    data_root : Path
        Folder holding the ANPHY-Sleep release (``.../ANPHY-Sleep/osfstorage``).
    cache_dir : Path
        Folder for the conditioned windows of every subject.
    runs_dir : Path
        Folder that receives one sub-folder per run.
    """

    data_root: Path
    cache_dir: Path
    runs_dir: Path


@dataclass
class DataConfig:
    """How recordings are conditioned and cut into windows and segments.

    Parameters
    ----------
    sfreq : float
        Sampling rate in Hz. Recordings are never resampled.
    window_s : float
        Length of one classified window in seconds.
    n_segments : int
        Segments per window; one graph is built per segment.
    highpass, lowpass, notch : float
        Filter frequencies in Hz.
    filter : str
        ``bandpass`` (``highpass``-``lowpass``, the original), ``highpass``
        (no low-pass: the full band up to Nyquist) or ``none`` (no notch and
        no band-pass, only the common average).
    notch_harmonics : int
        Notch the line frequency and its harmonics up to this multiple (1 is
        the original single notch); ignored with ``filter: none``.
    channel_subset : list of str
        Electrodes kept from the montage, empty for all; the common average
        is then taken over the kept electrodes.
    block_s : float
        Length of each continuous block that is filtered at once, in seconds.
    pad_s : float
        Padding filtered on both sides of a block and then discarded, in seconds.
    blocks_per_subject : int
        Blocks spread evenly over the scored part of each night.
    align_blocks : bool
        Move every block start back to the onset of the scored epoch it falls
        in. With ``block_s`` a multiple of the 30 s epoch and ``window_s`` a
        divisor of it, the windows then tile the scored epochs and none is lost
        to straddling an epoch boundary. False keeps the original block starts.
    max_subjects : int
        Keep only the first subjects; 0 keeps all of them.
    stages : tuple of str
        Sleep stage names in label order.
    rolling : bool
        Rolling windows (Exp 13). The cache keeps each filtered block whole,
        with the stage, epoch and elapsed time of every second, and windows
        are cut from it when they are read: a window is the ``window_s``
        seconds before a "now" second, and its label is the stage of that
        second, so windows may cross a stage change. Each window also carries
        the elapsed time since the recording started. False keeps the fixed,
        non-overlapping windows of Exp 1-12.
    train_stride_s : int
        Rolling only. Training draws one window per ``train_stride_s``
        seconds of each block, with its "now" second moved by a random 0 to
        ``train_stride_s - 1`` seconds every epoch, so every second is seen
        over the epochs at the cost of fixed windows. This stride also
        defines the windows comparable with Exp 12 ("aligned": ending on a
        multiple of it, inside one scored epoch) and the subject split.
    val_stride_s, test_stride_s : int
        Rolling only. Spacing of the "now" seconds of the validation and
        test windows, without random shifts.
    pretrain_unlabeled : bool
        Rolling and scored epochs. Stage 1 also uses training-subject windows
        without a stage: rolling windows whose "now" second is unscored, or
        the lights-on (``L``) epochs of the hypnogram; it needs no labels.
    scored_epochs : bool
        Whole nights cut into the hypnogram's scored epochs (Exp 14 and the
        final reruns). Every hypnogram row that lies inside the recording is
        one window of ``window_s`` seconds, which must equal the scored epoch
        length, and its label is that row's stage, as the scorer gave it. The
        night is filtered in ``block_s`` chunks with ``pad_s`` of real signal
        on both sides, and the cache is read from disk on demand
        (memory-mapped) because a whole night is about 4.5 GB. Training
        drops epochs above ``artifact_uv``; validation and test keep every
        scored epoch. False keeps the blocks of Exp 1-13.
    artifact_uv : float
        Scored epochs only. Largest absolute amplitude, in microvolts, of a
        training epoch; louder epochs are left out of both training stages.
    scaler_stride : int
        Scored epochs only. The per-electrode robust scaling is fitted on
        every ``scaler_stride``-th sample of the training epochs.
    context_epochs : int
        Scored epochs only. Consecutive epochs per item. 1 classifies each
        epoch alone. Above 1, an item is a run of epochs, the model's
        ``context`` component reads them in order, and each epoch is
        classified from itself and the epochs before it in the item
        (Exp 14b): training items tile each night with a random offset per
        read and are labelled at every scored, clean epoch; validation and
        test items end at, and are labelled only at, each scored epoch, so
        every scored epoch is predicted once from at most
        ``context_epochs - 1`` earlier epochs.
    """

    sfreq: float
    window_s: float
    n_segments: int
    highpass: float
    lowpass: float
    notch: float
    filter: str
    notch_harmonics: int
    channel_subset: list
    block_s: float
    pad_s: float
    blocks_per_subject: int
    align_blocks: bool
    max_subjects: int
    stages: tuple
    rolling: bool
    train_stride_s: int
    val_stride_s: int
    test_stride_s: int
    pretrain_unlabeled: bool
    scored_epochs: bool
    artifact_uv: float
    scaler_stride: int
    context_epochs: int

    @property
    def segment_len(self) -> int:
        """Samples per segment.

        Returns
        -------
        int
            ``window_s * sfreq / n_segments``.
        """
        return int(self.window_s * self.sfreq) // self.n_segments


@dataclass
class SplitConfig:
    """How many subjects go to validation and test.

    Parameters
    ----------
    n_val_subjects : int
        Validation subjects.
    n_test_subjects : int
        Test subjects; every other subject is used for training.
    n_folds : int
        0 for the single split above. Otherwise the test subjects stay the
        same and the remaining subjects are divided into this many folds,
        each used once as the validation set; ``n_val_subjects`` is then unused.
    fold : int
        Validation fold of this run, from 0 to ``n_folds - 1``. The run name
        gets ``_fold<fold>`` appended.
    fold_seed : bool
        Seed the model initialisation and training of each fold with
        ``seed + fold`` instead of ``seed``. The split itself always uses
        ``seed``, so the folds hold the same subjects either way. With one
        seed for every fold, all folds start from the same random weights,
        and structure left over from that draw (Exp 10's segment graph kept a
        correlation of 0.95-0.97 with it) looks reproducible across folds.
        In a rotation the offset is ``test_group`` instead.
    n_test_groups : int
        0 for the splits above. Otherwise every subject is tested once: the
        subjects are shuffled with ``seed`` and cut into this many groups;
        group ``test_group`` is the test set, the next group (cyclically) the
        validation set and the others the training set. ``n_val_subjects``,
        ``n_test_subjects`` and ``n_folds`` are then unused, and ``n_folds``
        must be 0. The run name gets ``_rot<test_group>`` appended.
    test_group : int
        Test group of this run, from 0 to ``n_test_groups - 1``.
    """

    n_val_subjects: int
    n_test_subjects: int
    n_folds: int
    fold: int
    fold_seed: bool
    n_test_groups: int
    test_group: int


@dataclass
class ModelConfig:
    """Widths of the model and the component chosen for each slot.

    Every component is a mapping with a ``name`` that selects a registered
    implementation in ``components.py``; its other keys are passed to that
    implementation.

    Parameters
    ----------
    d_model : int
        Width of a node representation after message passing.
    d_graph : int
        Width of the space the graphs are scored in.
    graph_learner, graph_conv, position, sequence, decoder, head, readout, context : dict
        Component specifications. ``readout`` turns the sequence outputs into
        one vector per window, and ``context`` reads those vectors across
        consecutive epochs before the head (``none`` passes them through).
    """

    d_model: int
    d_graph: int
    graph_learner: dict
    graph_conv: dict
    position: dict
    sequence: dict
    decoder: dict
    head: dict
    readout: dict
    context: dict


@dataclass
class LoaderConfig:
    """Batching of the windows.

    Parameters
    ----------
    batch_size : int
        Windows per batch.
    num_workers : int
        Loader worker processes. Values above 0 change the random stream of
        the augmentations, so runs are no longer bit-identical to the notebooks.
    """

    batch_size: int
    num_workers: int


@dataclass
class AugmentConfig:
    """Label-preserving perturbations applied to training windows.

    Parameters
    ----------
    time_roll : int
        Maximum circular shift in samples.
    scale : float
        Maximum relative amplitude change.
    channel_drop : float
        Probability of zeroing each electrode.
    """

    time_roll: int
    scale: float
    channel_drop: float


@dataclass
class PretrainConfig:
    """Stage 1: masked reconstruction.

    Parameters
    ----------
    epochs : int
        Maximum passes over the training windows; 0 skips stage 1.
    lr, weight_decay : float
        AdamW settings.
    mask_ratio : float
        Share of cells, or of electrodes, that are hidden.
    channel_mask_prob : float
        Share of windows that hide whole electrodes instead of random cells.
    mask_mode : str
        ``random``: whole electrodes are hidden independently (the original);
        ``cluster``: each hidden electrode takes its ``cluster_size`` nearest
        neighbours with it, so a hidden region cannot be copied from its
        neighbours (volume conduction).
    cluster_size : int
        Neighbours hidden with each centre electrode in ``cluster`` mode.
    patience : int
        Epochs without validation gain before stopping.
    val_every : int
        Epochs between validation passes.
    delta_penalty : float
        Weight of the L1 penalty on the change ``Delta``.
    graph_penalty : float
        Weight of the components' own regularisation terms (the KL of a learned
        segment graph from a uniform prior); 0 adds nothing.
    spectral_loss : bool
        Add a multi-resolution STFT loss on the masked electrodes (Exp 13).
    stft_sizes : tuple of int
        FFT sizes of that loss, in samples.
    future_steps : int
        Add a contrastive "predict the future" loss (CPC, van den Oord et al.
        2018): the output of segment ``t`` predicts the tokens of segments
        ``t+1 .. t+future_steps``. 0 adds nothing.
    future_temperature : float
        Temperature of that loss.
    uncertainty_weighting : bool
        Weigh the stage-1 losses with learned uncertainties (Kendall et al.,
        CVPR 2018). False adds them with the fixed weights below.
    reconstruction_weight, spectral_weight, future_weight : float
        Fixed weight of each stage-1 loss, applied before the uncertainty
        weighting if that is on; 1 everywhere is the plain sum used up to
        Exp 14.
    relative_losses : bool
        Divide every stage-1 loss of a batch by what a trivial prediction
        scores on it (detached): the reconstruction by predicting zeros on
        the same hidden cells, the spectral loss by the spectrum of the
        signal with its hidden cells set to zero, the contrastive loss by
        ``log`` of its number of candidates (chance). Each loss then reads
        as a share of its trivial baseline, starts near 1, and high-amplitude
        subjects or artifact epochs do not dominate it (Exp 15). Model
        selection uses the relative reconstruction loss. False keeps the
        absolute losses.
    spectral_floor : float
        Floor added inside the logarithms of the spectral loss, relative to
        the mean target magnitude of each electrode and FFT size, on top of
        the absolute 1e-5. It bounds the gradient of the log-magnitude term,
        which grows as one over the predicted magnitude (Exp 15); 0 keeps
        the absolute floor alone.
    max_skip_share : float
        Largest share of non-finite steps an epoch may have before the run
        stops with an error instead of continuing on the remaining steps
        (Exp 15); 1 never stops.
    init_run : str
        Skip stage 1 and start from the stage-1 weights of this earlier
        experiment, read from ``<runs_dir>/<init_run>[_fold<k>]/pretrain_best.pt``
        (same fold). Only the pretrained parts are loaded; ``readout``,
        ``context`` and ``head`` start fresh. Empty pretrains as usual.
    """

    epochs: int
    lr: float
    weight_decay: float
    mask_ratio: float
    channel_mask_prob: float
    mask_mode: str
    cluster_size: int
    patience: int
    val_every: int
    delta_penalty: float
    graph_penalty: float
    spectral_loss: bool
    stft_sizes: tuple
    future_steps: int
    future_temperature: float
    uncertainty_weighting: bool
    reconstruction_weight: float
    spectral_weight: float
    future_weight: float
    relative_losses: bool
    spectral_floor: float
    max_skip_share: float
    init_run: str


@dataclass
class FinetuneConfig:
    """Stage 2: sleep staging, linear probe first and then the whole network.

    Parameters
    ----------
    batch_size : int
        Windows per batch in stage 2 and in the test; 0 uses
        ``loader.batch_size``, which then also batches stage 1. The
        continuous decoder caps stage 1 at 4 windows of 30 s, while stage 2
        never runs it.
    epochs : int
        Maximum passes over the training windows.
    lr, weight_decay : float
        AdamW settings for the head.
    warmup_epochs : int
        Linear warm-up before the cosine decay.
    patience : int
        Epochs without validation gain before stopping.
    ema_decay : float
        Decay of the weight average used for model selection.
    freeze_epochs : int
        Epochs during which only the head is trained.
    backbone_lr_scale : float
        Backbone learning rate relative to ``lr`` once it is unfrozen.
    label_smoothing : float
        Label smoothing of the cross entropy.
    balanced_sampling : bool
        Oversample rare stages with inverse square-root frequency weights.
    class_balanced_loss : bool
        Weight the loss with effective-number class weights.
    class_balance_beta : float
        Beta of the effective-number weights.
    class_weight_min, class_weight_max : float
        Bounds of the class weights.
    focal_gamma : float
        Focal modulation exponent; 0 disables it.
    delta_penalty : float
        Weight of the L1 penalty on the change ``Delta``.
    graph_penalty : float
        Weight of the components' own regularisation terms; 0 adds nothing.
    """

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    ema_decay: float
    freeze_epochs: int
    backbone_lr_scale: float
    label_smoothing: float
    balanced_sampling: bool
    class_balanced_loss: bool
    class_balance_beta: float
    class_weight_min: float
    class_weight_max: float
    focal_gamma: float
    delta_penalty: float
    graph_penalty: float


@dataclass
class EvaluationConfig:
    """Controls and figures computed after training.

    Parameters
    ----------
    shuffle_trials : int
        Random segment orders averaged in the segment-order control.
    reconstruction_batches : int
        Validation batches scored in the reconstruction report.
    top_k : int
        Edges drawn per electrode and per sign in the 3-D graph figures: the
        ``top_k`` strongest positive and the ``top_k`` strongest negative.
    plot_segments : int
        Segments whose ``Delta_t`` is drawn, evenly spaced over the window;
        0 draws every segment.
    plots : bool
        Whether to write the figures.
    curve_batches : int
        Batches of training epochs scored after every epoch exactly as the
        validation epochs are (eval mode, no augmentation, fixed masks), for
        every stage-1 loss and for the stage-2 loss and scores, so the
        training and validation curves are comparable (Exp 15). 0 keeps the
        running training loss alone.
    nearest_candidates : int
        Nearest electrodes searched, closest first, by the ``nearest_electrode``
        baseline of the reconstruction report for a visible electrode to copy.
        With 4, the original setting, a hidden electrode whose four nearest
        electrodes are all hidden as well copies electrode 0 instead (about
        ``mask_ratio ** 4`` of the hidden cells), which understates the
        baseline. A value of at least ``C - 1`` searches every electrode.
    """

    shuffle_trials: int
    reconstruction_batches: int
    top_k: int
    plot_segments: int
    plots: bool
    curve_batches: int
    nearest_candidates: int


@dataclass
class Config:
    """Every setting of one run, resolved from the YAML layers.

    Parameters
    ----------
    experiment : str
        Run name; it names the output folder.
    description : str
        One line on what the experiment changes.
    seed : int
        Seed of every random draw.
    paths, data, split, model, loader, augment, pretrain, finetune, evaluation
        The sections of ``configs/default.yaml``.
    """

    experiment: str
    description: str
    seed: int
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    loader: LoaderConfig
    augment: AugmentConfig
    pretrain: PretrainConfig
    finetune: FinetuneConfig
    evaluation: EvaluationConfig

    @property
    def run_dir(self) -> Path:
        """Output folder of this run.

        Returns
        -------
        Path
            ``paths.runs_dir / experiment``.
        """
        return self.paths.runs_dir / self.experiment

    @property
    def device(self) -> torch.device:
        """The fastest available device.

        Returns
        -------
        torch.device
            CUDA, then MPS, then CPU.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def to_dict(self) -> dict:
        """Plain nested dictionary of the config, suitable for YAML.

        Returns
        -------
        dict
            The config with paths as strings and tuples as lists.
        """
        return _plain(dataclasses.asdict(self))


def _plain(value: Any) -> Any:
    """Convert paths and tuples inside a nested structure to YAML-friendly types.

    Parameters
    ----------
    value : Any
        A nested structure of dictionaries, lists, tuples and scalars.

    Returns
    -------
    Any
        The same structure with ``Path`` as ``str`` and tuples as lists.
    """
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def read_yaml(path: Path) -> dict:
    """Read one YAML file into a dictionary.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict
        Its content; an empty file gives an empty dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` into a copy of ``base``.

    Nested mappings are merged key by key. A component mapping whose ``name``
    changes is replaced as a whole, so parameters of the old component never
    leak into the new one.

    Parameters
    ----------
    base : dict
        Lower layer.
    override : dict
        Higher layer; its values win.

    Returns
    -------
    dict
        A new merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        replaces_component = (isinstance(value, dict) and isinstance(current, dict)
                              and "name" in value and value.get("name") != current.get("name"))
        if isinstance(value, dict) and isinstance(current, dict) and not replaces_component:
            result[key] = merge(current, value)
        else:
            result[key] = value
    return result


def dotted(pairs: list[str]) -> dict:
    """Turn ``a.b.c=value`` strings into a nested dictionary.

    Values are parsed as YAML, so ``5e-4``, ``true`` and ``[1, 2]`` keep their types.

    Parameters
    ----------
    pairs : list of str
        Assignments from the command line.

    Returns
    -------
    dict
        The nested overrides.

    Raises
    ------
    ValueError
        If an assignment has no ``=``.
    """
    result: dict = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {pair!r}")
        node = result
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return result


def experiment_layers(path: Path, seen: tuple = ()) -> list[dict]:
    """Read an experiment file and, recursively, the files it is based on.

    An experiment file may name another one under ``base``, or a list of
    them, which are layered in that order (a model chain first, then a shared
    protocol, as the final reruns do); paths are relative to the file that
    names them.

    Parameters
    ----------
    path : Path
        Experiment file.
    seen : tuple
        Files already visited, to detect cycles.

    Returns
    -------
    list of dict
        Layers from the oldest base to ``path`` itself, without ``base`` keys.

    Raises
    ------
    ValueError
        If the ``base`` chain loops.
    """
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"config base chain loops at {path}")
    layer = read_yaml(path)
    base = layer.pop("base", None)
    bases = [] if not base else [base] if isinstance(base, str) else list(base)
    parents = [parent for name in bases
               for parent in experiment_layers(path.parent / name, seen + (path,))]
    return parents + [layer]


def find_experiment_file(reference: str | Path) -> Path:
    """Locate an experiment file from a path or a bare name.

    Parameters
    ----------
    reference : str or Path
        ``configs/exp5_signed_gcn.yaml``, ``exp5_signed_gcn.yaml`` or ``exp5_signed_gcn``.

    Returns
    -------
    Path
        The existing file.

    Raises
    ------
    FileNotFoundError
        If no candidate exists.
    """
    reference = Path(reference)
    candidates = [reference, CONFIG_DIR / reference, CONFIG_DIR / f"{reference}.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no experiment config for {str(reference)!r} "
                            f"(looked at {', '.join(map(str, candidates))})")


def load_run_config(run_dir: Path, overrides: dict) -> Config:
    """The configuration a finished run saved, with settings added since at their defaults.

    A run writes its full configuration to ``config.yaml``. Settings added to
    ``configs/default.yaml`` after the run are missing from that file; they are
    taken from the defaults, which by the repository's rule keep the old
    behaviour, so the run is rebuilt as it ran.

    Parameters
    ----------
    run_dir : Path
        Run folder with ``config.yaml``.
    overrides : dict
        Nested settings merged on top (empty for none).

    Returns
    -------
    Config
        The validated config.
    """
    layered = merge(merge(read_yaml(DEFAULTS), read_yaml(Path(run_dir) / "config.yaml")), overrides)
    return build(Config, layered)


def resolve(experiment_file: str | Path | None = None, overrides: list[dict] | None = None,
            name: str | None = None) -> Config:
    """Build a config from its layers.

    Layers, lowest first: ``configs/default.yaml``, the ``base`` chain of the
    experiment file, the experiment file, then each mapping in ``overrides``
    (the plan's ``common`` and ``set`` blocks, ``--quick``, ``--set``).

    Parameters
    ----------
    experiment_file : str or Path, optional
        Detailed experiment config; omitted means defaults plus overrides only.
    overrides : list of dict, optional
        Higher layers, lowest first.
    name : str, optional
        Run name; defaults to the ``experiment`` key, then the file name. With
        ``split.n_folds`` set, ``_fold<fold>`` is appended; with
        ``split.n_test_groups`` set, ``_rot<test_group>``.

    Returns
    -------
    Config
        The validated config.

    Raises
    ------
    ValueError
        If both ``split.n_folds`` and ``split.n_test_groups`` are set.
    """
    layers = [read_yaml(DEFAULTS)]
    if experiment_file is not None:
        experiment_file = find_experiment_file(experiment_file)
        layers += experiment_layers(experiment_file)
    layers += overrides or []

    merged: dict = {}
    for layer in layers:
        merged = merge(merged, layer)
    if name:
        merged["experiment"] = name
    elif experiment_file is not None and not merged.get("experiment"):
        merged["experiment"] = Path(experiment_file).stem
    split = merged.get("split") or {}
    if split.get("n_folds") and split.get("n_test_groups"):
        raise ValueError("split.n_folds and split.n_test_groups cannot both be set")
    if split.get("n_folds"):
        merged["experiment"] = f"{merged['experiment']}_fold{split.get('fold')}"
    if split.get("n_test_groups"):
        merged["experiment"] = f"{merged['experiment']}_rot{split.get('test_group')}"
    cfg = build(Config, merged)
    check_model(dataclasses.asdict(cfg.model))
    return cfg


def build(schema: type, data: dict, where: str = "config") -> Any:
    """Validate a nested dictionary against a dataclass schema and cast its values.

    Parameters
    ----------
    schema : type
        Dataclass to build.
    data : dict
        Values for its fields.
    where : str
        Dotted location, used in error messages.

    Returns
    -------
    Any
        An instance of ``schema``.

    Raises
    ------
    KeyError
        If a field is missing or an unknown key is present.
    """
    if not isinstance(data, dict):
        raise TypeError(f"{where} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(schema)
    names = [field.name for field in dataclasses.fields(schema)]
    unknown = sorted(set(data) - set(names))
    missing = [name for name in names if name not in data]
    if unknown:
        raise KeyError(f"unknown keys in {where}: {unknown}")
    if missing:
        raise KeyError(f"missing keys in {where}: {missing} (add them to {DEFAULTS})")
    return schema(**{name: cast(hints[name], data[name], f"{where}.{name}") for name in names})


def cast(kind: Any, value: Any, where: str) -> Any:
    """Convert one YAML value to the type its field declares.

    Parameters
    ----------
    kind : Any
        Declared field type.
    value : Any
        Value read from YAML.
    where : str
        Dotted location, used in error messages.

    Returns
    -------
    Any
        The converted value.

    Raises
    ------
    TypeError
        If the value cannot represent the declared type.
    """
    if dataclasses.is_dataclass(kind):
        return build(kind, value, where)
    if isinstance(kind, types.UnionType) or typing.get_origin(kind) is typing.Union:
        return value
    if kind is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where} must be true or false, got {value!r}")
        return value
    if kind is dict:
        if not isinstance(value, dict) or "name" not in value:
            raise TypeError(f"{where} must be a mapping with a 'name', got {value!r}")
        return dict(value)
    if kind is tuple:
        return tuple(value)
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{where} must be {kind.__name__}, got {value!r}") from error


def read_plan(path: Path = PLAN) -> tuple[dict, list[dict]]:
    """Read the high-level plan of experiments.

    Each entry of ``experiments`` is either a bare name, meaning
    ``configs/<name>.yaml``, or a mapping with ``name`` and optionally
    ``config`` (an experiment file), ``set`` (inline overrides) and
    ``enabled``.

    Parameters
    ----------
    path : Path
        Plan file.

    Returns
    -------
    common : dict
        Overrides applied to every experiment.
    entries : list of dict
        One normalised entry per experiment, with keys ``name``, ``config``,
        ``set`` and ``enabled``.

    Raises
    ------
    ValueError
        If two entries share a name.
    """
    plan = read_yaml(path)
    entries = []
    for raw in plan.get("experiments") or []:
        entry = {"name": raw} if isinstance(raw, str) else dict(raw)
        unknown = set(entry) - {"name", "config", "set", "enabled", "description"}
        if unknown:
            raise KeyError(f"unknown keys in plan entry {entry.get('name')!r}: {sorted(unknown)}")
        if "config" not in entry and (CONFIG_DIR / f"{entry['name']}.yaml").is_file():
            entry["config"] = entry["name"]
        set_block = entry.get("set") or {}
        if entry.get("description"):
            set_block = merge(set_block, {"description": entry["description"]})
        entries.append({"name": entry["name"], "config": entry.get("config"),
                        "set": set_block, "enabled": entry.get("enabled", True)})
    names = [entry["name"] for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate experiment names in {path}: {duplicates}")
    return plan.get("common") or {}, entries


def git_state() -> dict:
    """The commit the code was run from, so a run can be traced to its code.

    Returns
    -------
    dict
        ``commit`` and ``dirty`` (uncommitted changes), or ``None`` values
        outside a git repository.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
