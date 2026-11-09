from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import mne
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from config import Config

mne.set_log_level("ERROR")

Arrays = dict[str, tuple[np.ndarray, np.ndarray]]


def find_montage(root: Path) -> Path:
    """Locate the electrode position file anywhere under the dataset root.

    Parameters
    ----------
    root : Path
        Folder holding the ANPHY release.

    Returns
    -------
    Path
        The ``.pos`` file with electrode names and coordinates.

    Raises
    ------
    FileNotFoundError
        If no ``.pos`` file exists under ``root``.
    """
    hits = sorted(Path(root).rglob("*.pos"))
    if not hits:
        raise FileNotFoundError(f"no .pos montage file under {root} "
                                "(download the data with scripts/download_anphy.py)")
    return hits[0]


def read_montage(path: Path) -> tuple[list[str], np.ndarray]:
    """Read electrode names and 3-D coordinates from a ``.pos`` file.

    Lines with fewer than five fields, such as the electrode count header,
    are skipped.

    Parameters
    ----------
    path : Path
        Whitespace-separated file with columns index, name, x, y, z.

    Returns
    -------
    names : list of str
        Uppercase electrode names.
    coords : np.ndarray
        Coordinates ``[n_electrodes, 3]``.
    """
    names, coords = [], []
    for line in path.read_text().splitlines():
        parts = line.split()[1:]
        if len(parts) >= 4:
            names.append(parts[0].strip().upper())
            coords.append([float(v) for v in parts[1:4]])
    return names, np.asarray(coords, dtype=np.float32)


def find_subjects(root: Path) -> list[tuple[str, Path, Path]]:
    """Pair every EDF recording under the root with its hypnogram.

    Parameters
    ----------
    root : Path
        Folder with one sub-folder per subject.

    Returns
    -------
    list of tuple
        ``(subject_id, edf_path, hypnogram_path)`` sorted by subject; folders
        without both files are skipped.
    """
    found = []
    for edf in sorted(Path(root).rglob("*.edf")):
        txts = sorted(edf.parent.glob("*.txt"))
        if txts:
            found.append((edf.parent.name, edf, txts[0]))
    return found


def read_hypnogram(path: Path) -> np.ndarray:
    """Read a scored hypnogram into stage intervals.

    Parameters
    ----------
    path : Path
        Whitespace-separated file with columns stage, onset, duration.

    Returns
    -------
    np.ndarray
        Structured array with fields ``stage`` (str), ``onset`` and
        ``duration`` (seconds).
    """
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append((parts[0].strip().upper(), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return np.array(rows, dtype=[("stage", "U8"), ("onset", "f8"), ("duration", "f8")])


def preprocessing_signature(cfg: Config, channels: list[str]) -> str:
    """Hash every setting that changes the cached arrays.

    The payload is the one the notebooks used, so their cache files are reused.
    Settings added later join the payload only when they differ from the
    original behaviour, which keeps the old cache names valid. A rolling cache
    holds whole blocks, which do not depend on the window length, so
    ``window_s`` leaves its payload and windows of every length share it. A
    scored-epoch cache holds whole nights; the block settings only decide how
    the night is filtered, so ``blocks_per_subject`` and ``align_blocks``
    leave its payload.

    Parameters
    ----------
    cfg : Config
        Run settings.
    channels : list of str
        Electrode names from the montage.

    Returns
    -------
    str
        Eight hexadecimal characters that go into the cache file names.
    """
    data = cfg.data
    if data.scored_epochs:
        payload = (("scored_epochs", True), data.window_s, data.block_s, data.pad_s,
                   data.highpass, data.lowpass, data.notch, tuple(channels))
        if data.filter != "bandpass":
            payload += (("filter", data.filter),)
        if data.notch_harmonics != 1:
            payload += (("notch_harmonics", data.notch_harmonics),)
        return hashlib.sha256(repr(payload).encode()).hexdigest()[:8]
    payload = (data.blocks_per_subject, data.block_s, data.pad_s, data.window_s,
               data.highpass, data.lowpass, data.notch, tuple(channels))
    if data.rolling:
        payload = (data.blocks_per_subject, data.block_s, data.pad_s, data.highpass,
                   data.lowpass, data.notch, tuple(channels), ("rolling", True))
    if data.align_blocks:
        payload += (("align_blocks", True),)
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:8]


def match_channels(raw_names: list[str], wanted: list[str]) -> dict[str, str]:
    """Map EDF channel labels such as ``EEG Fp1-REF`` onto montage names.

    Parameters
    ----------
    raw_names : list of str
        Channel names in the EDF file.
    wanted : list of str
        Uppercase electrode names from the montage.

    Returns
    -------
    dict
        Rename mapping from EDF label to montage name.
    """
    patterns = {
        name: re.compile(rf"^(?:EEG[\s:_-]*)?{re.escape(name)}"
                         rf"(?:[\s:_-]*(?:REF|AVG|LE))?[\s:_-]*$", re.IGNORECASE)
        for name in wanted
    }
    mapping = {}
    for raw_name in raw_names:
        clean = raw_name.strip().upper()
        for name, pattern in patterns.items():
            if pattern.fullmatch(clean):
                if raw_name != name:
                    mapping[raw_name] = name
                break
    return mapping


def window_label(hypnogram: np.ndarray, start_s: float, length_s: float,
                 stages: tuple[str, ...]) -> int:
    """Assign a stage to a window that lies entirely inside one scored epoch.

    Parameters
    ----------
    hypnogram : np.ndarray
        Stage intervals from :func:`read_hypnogram`.
    start_s : float
        Window onset in seconds from the recording start.
    length_s : float
        Window length in seconds.
    stages : tuple of str
        Stage names in label order.

    Returns
    -------
    int
        Stage index, or ``-1`` if the window straddles a transition or falls
        in an unscored stage.
    """
    stop_s = start_s + length_s
    inside = ((hypnogram["onset"] <= start_s) &
              (hypnogram["onset"] + hypnogram["duration"] >= stop_s))
    if not inside.any():
        return -1
    stage = hypnogram["stage"][np.argmax(inside)]
    return stages.index(stage) if stage in stages else -1


def load_subject(cfg: Config, channels: list[str], subject: str, edf_path: Path,
                 hyp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Condition one recording and return its labelled windows, using the cache.

    Blocks are spread evenly over the scored part of the night, optionally
    moved back to the onset of their scored epoch (``data.align_blocks``), and
    filtered with padding so no window carries a filter edge (notch, band-pass,
    average reference, no resampling). Windows with amplitudes above 500 uV
    are dropped. The cache file is written under a temporary name and then
    renamed, so jobs building the same cache at once never read a half-written
    file.

    Parameters
    ----------
    cfg : Config
        Run settings.
    channels : list of str
        Electrode names from the montage.
    subject : str
        Subject id, used in the cache file name.
    edf_path : Path
        EDF recording.
    hyp_path : Path
        Matching hypnogram.

    Returns
    -------
    X : np.ndarray
        Windows ``[n, C, samples]`` in float16 microvolts.
    y : np.ndarray
        Stage indices ``[n]``.
    """
    data = cfg.data
    cache = cfg.paths.cache_dir / f"{subject}_{preprocessing_signature(cfg, channels)}.npz"
    if cache.exists():
        blob = np.load(cache)
        return blob["X"], blob["y"]

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.rename_channels(match_channels(raw.ch_names, channels))
    keep = [c for c in channels if c in raw.ch_names]
    raw.pick(keep)

    hypnogram = read_hypnogram(hyp_path)
    sfreq = float(raw.info["sfreq"])
    window_n = int(data.window_s * sfreq)
    block_n, pad_n = int(data.block_s * sfreq), int(data.pad_s * sfreq)

    scored_start = float(hypnogram["onset"].min())
    scored_stop = min(float((hypnogram["onset"] + hypnogram["duration"]).max()),
                      raw.n_times / sfreq)
    starts = np.linspace(scored_start, max(scored_start, scored_stop - data.block_s),
                         data.blocks_per_subject)
    if data.align_blocks:
        onsets = np.sort(hypnogram["onset"])
        starts = np.unique(onsets[np.searchsorted(onsets, starts, side="right") - 1])

    windows, labels = [], []
    for block_start in starts:
        lo = max(0, int(block_start * sfreq) - pad_n)
        hi = min(raw.n_times, int(block_start * sfreq) + block_n + pad_n)
        signal = raw.get_data(start=lo, stop=hi) * 1e6

        signal = mne.filter.notch_filter(signal, sfreq, [data.notch], verbose=False)
        signal = mne.filter.filter_data(signal, sfreq, data.highpass, data.lowpass, verbose=False)
        signal = signal - signal.mean(axis=0, keepdims=True)

        offset = int(block_start * sfreq) - lo
        for k in range(int(data.block_s // data.window_s)):
            a = offset + k * window_n
            if a + window_n > signal.shape[1]:
                break
            label = window_label(hypnogram, block_start + k * data.window_s, data.window_s,
                                 data.stages)
            chunk = signal[:, a:a + window_n]
            if label < 0 or np.abs(chunk).max() > 500:
                continue
            windows.append(chunk.astype(np.float16))
            labels.append(label)

    raw.close()
    X = np.stack(windows) if windows else np.zeros((0, len(keep), window_n), np.float16)
    y = np.asarray(labels, dtype=np.int64)
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{cache.stem}.{os.getpid()}.partial.npz")
    np.savez_compressed(partial, X=X, y=y)
    os.replace(partial, cache)
    return X, y


def load_dataset(cfg: Config) -> tuple[Arrays, list[str], np.ndarray]:
    """Load the windows of every subject, building the cache where it is missing.

    Parameters
    ----------
    cfg : Config
        Run settings.

    Returns
    -------
    arrays : dict
        ``subject -> (X, y)``. With ``data.rolling``: ``subject -> (blocks,
        y, cache)``, where ``blocks`` is ``[n_blocks, C, samples]``, ``y`` the
        labels of the reference windows (:func:`reference_labels`, used by the
        subject split) and ``cache`` everything :func:`load_subject_blocks`
        returns. With ``data.scored_epochs``: ``subject -> (X, y, epochs)``,
        where ``X`` is the memory-mapped ``[n_epochs, C, samples]``, ``y`` the
        stages of the scored epochs (used by the subject split) and
        ``epochs`` everything :func:`load_subject_epochs` returns.
    channels : list of str
        Electrode names.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.
    """
    channels, coords = read_montage(find_montage(cfg.paths.data_root))
    if cfg.data.channel_subset:
        wanted = [name.upper() for name in cfg.data.channel_subset]
        missing = sorted(set(wanted) - set(channels))
        if missing:
            raise ValueError(f"data.channel_subset names electrodes not in the montage: {missing}")
        keep = [i for i, name in enumerate(channels) if name in wanted]
        channels, coords = [channels[i] for i in keep], coords[keep]
    if (cfg.data.filter != "bandpass" or cfg.data.notch_harmonics != 1) and not cfg.data.scored_epochs:
        raise ValueError("data.filter and data.notch_harmonics apply to data.scored_epochs only")
    subjects = find_subjects(cfg.paths.data_root)
    if cfg.data.max_subjects:
        subjects = subjects[:cfg.data.max_subjects]

    if cfg.data.scored_epochs and cfg.data.rolling:
        raise ValueError("data.scored_epochs and data.rolling exclude each other")
    if cfg.data.context_epochs > 1 and not cfg.data.scored_epochs:
        raise ValueError("data.context_epochs above 1 needs data.scored_epochs")
    arrays: Arrays = {}
    bar = tqdm(subjects, desc="subjects")
    for subject, edf_path, hyp_path in bar:
        if cfg.data.scored_epochs:
            epochs = load_subject_epochs(cfg, channels, subject, edf_path, hyp_path)
            y = epochs["stage"][epochs["stage"] >= 0].astype(np.int64)
            bar.write(epoch_summary(cfg, subject, epochs))
            if len(y):
                arrays[subject] = (epochs["X"], y, epochs)
            continue
        if cfg.data.rolling:
            blocks = load_subject_blocks(cfg, channels, subject, edf_path, hyp_path)
            y = reference_labels(cfg, blocks)
            if len(y):
                arrays[subject] = (blocks["X"], y, blocks)
            continue
        X, y = load_subject(cfg, channels, subject, edf_path, hyp_path)
        if len(y):
            arrays[subject] = (X, y)
    if cfg.data.scored_epochs:
        stage = np.concatenate([arrays[s][2]["stage"] for s in arrays])
        peak = np.concatenate([arrays[s][2]["peak"] for s in arrays])
        counts = np.bincount(stage[stage >= 0], minlength=len(cfg.data.stages))
        print(f"whole nights: {len(arrays)} subjects, {len(stage)} epochs, "
              f"{int((stage >= 0).sum())} scored, "
              f"{int(((stage >= 0) & (peak <= cfg.data.artifact_uv)).sum())} scored and clean, "
              + " ".join(f"{name} {n}" for name, n in zip(cfg.data.stages, counts)))
    return arrays, channels, coords


def epoch_summary(cfg: Config, subject: str, epochs: dict[str, np.ndarray]) -> str:
    """One log line on a whole night: how many epochs it holds and of which kind.

    Printed for every subject as it loads, so a run's log shows that whole
    nights are read, not blocks of them.

    Parameters
    ----------
    cfg : Config
        Run settings (``data.window_s``, ``data.artifact_uv``, ``data.stages``).
    subject : str
        Subject id.
    epochs : dict
        Output of :func:`load_subject_epochs`.

    Returns
    -------
    str
        Subject, epochs, scored, scored and clean, unscored, hours and the
        epochs per stage.
    """
    stage, peak = epochs["stage"], epochs["peak"]
    scored = stage >= 0
    counts = np.bincount(stage[scored], minlength=len(cfg.data.stages))
    return (f"{subject}: {len(stage)} epochs ({len(stage) * cfg.data.window_s / 3600:.2f} h), "
            f"{int(scored.sum())} scored, {int((scored & (peak <= cfg.data.artifact_uv)).sum())} "
            f"scored and clean, {int((~scored).sum())} unscored | "
            + " ".join(f"{name} {n}" for name, n in zip(cfg.data.stages, counts)))


def load_subject_blocks(cfg: Config, channels: list[str], subject: str, edf_path: Path,
                        hyp_path: Path) -> dict[str, np.ndarray]:
    """Condition one recording into whole blocks with per-second labels, using the cache.

    The blocks are placed and filtered exactly as in :func:`load_subject`, but
    kept whole, so rolling windows (``data.rolling``) can be cut from them at
    any second. Every second of every block gets its stage, the scored epoch
    it lies in, the time since the recording started at its end, and its
    largest amplitude, which the windows use to drop artifacts above
    500 uV as before.

    Parameters
    ----------
    cfg : Config
        Run settings.
    channels : list of str
        Electrode names from the montage.
    subject : str
        Subject id, used in the cache file name.
    edf_path : Path
        EDF recording.
    hyp_path : Path
        Matching hypnogram.

    Returns
    -------
    dict
        ``X`` ``[n_blocks, C, block samples]`` float16 microvolts; per second
        ``[n_blocks, block seconds]``: ``stage`` (index, -1 when not a scored
        stage), ``epoch`` (row of the hypnogram, -1 when none), ``elapsed``
        (seconds since the recording started, at the end of the second) and
        ``peak`` (largest absolute amplitude, infinite where the recording ended).
    """
    data = cfg.data
    cache = (cfg.paths.cache_dir
             / f"{subject}_{preprocessing_signature(cfg, channels)}_blocks.npz")
    if cache.exists():
        blob = np.load(cache)
        return {key: blob[key] for key in blob.files}

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.rename_channels(match_channels(raw.ch_names, channels))
    keep = [c for c in channels if c in raw.ch_names]
    raw.pick(keep)

    hypnogram = read_hypnogram(hyp_path)
    sfreq = float(raw.info["sfreq"])
    block_n, pad_n, second_n = int(data.block_s * sfreq), int(data.pad_s * sfreq), int(sfreq)
    n_seconds = int(data.block_s)

    scored_start = float(hypnogram["onset"].min())
    scored_stop = min(float((hypnogram["onset"] + hypnogram["duration"]).max()),
                      raw.n_times / sfreq)
    starts = np.linspace(scored_start, max(scored_start, scored_stop - data.block_s),
                         data.blocks_per_subject)
    if data.align_blocks:
        onsets = np.sort(hypnogram["onset"])
        starts = np.unique(onsets[np.searchsorted(onsets, starts, side="right") - 1])

    stage_index = {name: index for index, name in enumerate(data.stages)}
    blocks, stage, epoch, elapsed, peak = [], [], [], [], []
    for block_start in starts:
        lo = max(0, int(block_start * sfreq) - pad_n)
        hi = min(raw.n_times, int(block_start * sfreq) + block_n + pad_n)
        signal = raw.get_data(start=lo, stop=hi) * 1e6
        signal = mne.filter.notch_filter(signal, sfreq, [data.notch], verbose=False)
        signal = mne.filter.filter_data(signal, sfreq, data.highpass, data.lowpass, verbose=False)
        signal = signal - signal.mean(axis=0, keepdims=True)

        offset = int(block_start * sfreq) - lo
        block = np.zeros((len(keep), block_n), dtype=np.float32)
        available = max(0, min(block_n, signal.shape[1] - offset))
        block[:, :available] = signal[:, offset:offset + available]

        seconds = block_start + np.arange(n_seconds)
        rows = np.full(n_seconds, -1, dtype=np.int32)
        for j, second in enumerate(seconds):
            inside = ((hypnogram["onset"] <= second)
                      & (hypnogram["onset"] + hypnogram["duration"] >= second + 1))
            if inside.any():
                rows[j] = int(np.argmax(inside))
        labels = np.array([stage_index.get(hypnogram["stage"][row], -1) if row >= 0 else -1
                           for row in rows], dtype=np.int8)
        amplitude = np.abs(block[:, :n_seconds * second_n]).reshape(
            len(keep), n_seconds, second_n).max(axis=(0, 2))
        amplitude[np.arange(n_seconds) * second_n + second_n > available] = np.inf

        blocks.append(block.astype(np.float16))
        stage.append(labels)
        epoch.append(rows)
        elapsed.append((seconds + 1).astype(np.float32))
        peak.append(amplitude.astype(np.float32))

    raw.close()
    result = {"X": np.stack(blocks), "stage": np.stack(stage), "epoch": np.stack(epoch),
              "elapsed": np.stack(elapsed), "peak": np.stack(peak)}
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{cache.stem}.{os.getpid()}.partial.npz")
    np.savez_compressed(partial, **result)
    os.replace(partial, cache)
    return result


def condition(signal: np.ndarray, sfreq: float, data) -> np.ndarray:
    """Filter one block of a night as ``data.filter`` asks, before the common average.

    ``bandpass`` is the original conditioning (one notch at ``data.notch``, then
    ``highpass``-``lowpass``); ``highpass`` notches the line frequency and its
    harmonics up to ``notch_harmonics`` and only removes drift below
    ``highpass``, keeping the band up to Nyquist; ``none`` returns the signal
    as recorded.

    Parameters
    ----------
    signal : np.ndarray
        ``[C, samples]`` in microvolts.
    sfreq : float
        Sampling rate in Hz.
    data : DataConfig
        ``filter``, ``notch``, ``notch_harmonics``, ``highpass``, ``lowpass``.

    Returns
    -------
    np.ndarray
        The filtered block.

    Raises
    ------
    ValueError
        If ``data.filter`` is unknown.
    """
    if data.filter == "none":
        return signal
    if data.filter not in ("bandpass", "highpass"):
        raise ValueError(f"data.filter must be bandpass, highpass or none, not {data.filter!r}")
    lines = [data.notch * k for k in range(1, data.notch_harmonics + 1) if data.notch * k < sfreq / 2]
    signal = mne.filter.notch_filter(signal, sfreq, lines, verbose=False)
    lowpass = data.lowpass if data.filter == "bandpass" else None
    return mne.filter.filter_data(signal, sfreq, data.highpass, lowpass, verbose=False)


def load_subject_epochs(cfg: Config, channels: list[str], subject: str, edf_path: Path,
                        hyp_path: Path) -> dict[str, np.ndarray]:
    """Condition one whole night into its scored epochs, using the cache.

    Every hypnogram row that lies inside the recording becomes one epoch of
    ``data.window_s`` seconds starting at the row's onset. The night is
    filtered as before (notch, band-pass, common average) in chunks of
    ``data.block_s`` seconds with ``data.pad_s`` of real signal on both sides,
    which covers the filters' transients, and is never resampled. The signal
    goes to an uncompressed ``.npy`` that is memory-mapped when read, since a
    whole night does not fit in memory for every subject at once.

    Parameters
    ----------
    cfg : Config
        Run settings.
    channels : list of str
        Electrode names from the montage.
    subject : str
        Subject id, used in the cache file names.
    edf_path : Path
        EDF recording.
    hyp_path : Path
        Matching hypnogram.

    Returns
    -------
    dict
        ``X`` ``[n_epochs, C, epoch samples]`` float16 microvolts (read-only
        memory map); per epoch: ``stage`` (index, -1 for lights on and any
        other unscored row), ``onset`` (seconds) and ``peak`` (largest
        absolute amplitude in microvolts).

    Raises
    ------
    ValueError
        If the recording's sampling rate is not ``data.sfreq``, or the
        hypnogram rows are not consecutive epochs of ``data.window_s`` seconds.
    """
    data = cfg.data
    stem = f"{subject}_{preprocessing_signature(cfg, channels)}_epochs"
    signal_path = cfg.paths.cache_dir / f"{stem}.npy"
    meta_path = cfg.paths.cache_dir / f"{stem}_meta.npz"
    if signal_path.exists() and meta_path.exists():
        meta = np.load(meta_path)
        return {"X": np.load(signal_path, mmap_mode="r"),
                **{key: meta[key] for key in meta.files}}

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.rename_channels(match_channels(raw.ch_names, channels))
    keep = [c for c in channels if c in raw.ch_names]
    raw.pick(keep)
    sfreq = float(raw.info["sfreq"])
    if sfreq != data.sfreq:
        raise ValueError(f"{subject}: recorded at {sfreq} Hz, expected {data.sfreq} Hz; "
                         "recordings are never resampled")

    hypnogram = read_hypnogram(hyp_path)
    hypnogram = hypnogram[np.argsort(hypnogram["onset"], kind="stable")]
    epoch_n, pad_n = int(round(data.window_s * sfreq)), int(data.pad_s * sfreq)
    starts = np.round(hypnogram["onset"] * sfreq).astype(np.int64)
    inside = (starts >= 0) & (starts + epoch_n <= raw.n_times)
    hypnogram, starts = hypnogram[inside], starts[inside]
    if not (np.all(hypnogram["duration"] == data.window_s) and np.all(np.diff(starts) == epoch_n)):
        raise ValueError(f"{subject}: hypnogram rows are not consecutive {data.window_s} s epochs")

    stage_index = {name: index for index, name in enumerate(data.stages)}
    stage = np.array([stage_index.get(name, -1) for name in hypnogram["stage"]], dtype=np.int8)
    peak = np.empty(len(starts), dtype=np.float32)
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cfg.paths.cache_dir / f"{stem}.{os.getpid()}.partial.npy"
    X = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                  shape=(len(starts), len(keep), epoch_n))
    rows_per_block = max(1, int(data.block_s // data.window_s))
    for first in tqdm(range(0, len(starts), rows_per_block), desc=subject, leave=False):
        last = min(first + rows_per_block, len(starts))
        begin, end = int(starts[first]), int(starts[last - 1]) + epoch_n
        lo, hi = max(0, begin - pad_n), min(raw.n_times, end + pad_n)
        signal = condition(raw.get_data(start=lo, stop=hi) * 1e6, sfreq, data)
        signal = signal - signal.mean(axis=0, keepdims=True)
        epochs = signal[:, begin - lo:end - lo].reshape(len(keep), last - first, epoch_n)
        epochs = epochs.transpose(1, 0, 2)
        X[first:last] = epochs.astype(np.float16)
        peak[first:last] = np.abs(epochs).max(axis=(1, 2))
    raw.close()
    X.flush()
    del X
    os.replace(partial, signal_path)

    meta = {"stage": stage, "onset": hypnogram["onset"].astype(np.float64), "peak": peak}
    partial_meta = cfg.paths.cache_dir / f"{stem}_meta.{os.getpid()}.partial.npz"
    np.savez(partial_meta, **meta)
    os.replace(partial_meta, meta_path)
    return {"X": np.load(signal_path, mmap_mode="r"), **meta}


def window_ends(cfg: Config, blocks: dict[str, np.ndarray], length_s: int,
                labelled: bool) -> tuple[np.ndarray, np.ndarray]:
    """Every usable "now" second of every block, for windows of a given length.

    A window ends at second ``e`` of its block and covers seconds
    ``e - length_s .. e - 1``; it is usable when none of them exceeds 500 uV
    and, if ``labelled``, when its last second has a stage.

    Parameters
    ----------
    cfg : Config
        Run settings.
    blocks : dict
        Output of :func:`load_subject_blocks`.
    length_s : int
        Window length in seconds.
    labelled : bool
        Require a stage at the last second.

    Returns
    -------
    block : np.ndarray
        Block index of every usable window.
    end : np.ndarray
        Its end second, between ``length_s`` and the block length.
    """
    peak, stage = blocks["peak"], blocks["stage"]
    n_blocks, n_seconds = peak.shape
    found_block, found_end = [], []
    for b in range(n_blocks):
        for end in range(length_s, n_seconds + 1):
            if peak[b, end - length_s:end].max() > 500:
                continue
            if labelled and stage[b, end - 1] < 0:
                continue
            found_block.append(b)
            found_end.append(end)
    return np.asarray(found_block, dtype=np.int64), np.asarray(found_end, dtype=np.int64)


def inside_one_epoch(blocks: dict[str, np.ndarray], block: np.ndarray, end: np.ndarray,
                     length_s: int) -> np.ndarray:
    """Whether each window lies inside a single scored epoch.

    Parameters
    ----------
    blocks : dict
        Output of :func:`load_subject_blocks`.
    block, end : np.ndarray
        Windows, as from :func:`window_ends`.
    length_s : int
        Window length in seconds.

    Returns
    -------
    np.ndarray
        Boolean per window.
    """
    epoch = blocks["epoch"]
    first = epoch[block, end - length_s]
    last = epoch[block, end - 1]
    return (first == last) & (last >= 0)


def reference_labels(cfg: Config, blocks: dict[str, np.ndarray]) -> np.ndarray:
    """Labels of the windows a fixed-window run would have had, for the subject split.

    These are the ``train_stride_s``-second windows that end on a multiple of
    that stride and lie inside one scored epoch, which for a 5 s stride are
    exactly Exp 12's windows. The split draws on them, so rolling runs get the
    same subjects in every fold whatever their window length.

    Parameters
    ----------
    cfg : Config
        Run settings (``data.train_stride_s``).
    blocks : dict
        Output of :func:`load_subject_blocks`.

    Returns
    -------
    np.ndarray
        Stage indices of those windows.
    """
    stride = cfg.data.train_stride_s
    block, end = window_ends(cfg, blocks, stride, labelled=True)
    keep = (end % stride == 0) & inside_one_epoch(blocks, block, end, stride)
    return blocks["stage"][block[keep], end[keep] - 1].astype(np.int64)


class RollingDataset(Dataset):
    """Rolling windows of several subjects: the seconds before a "now" second (Exp 13).

    Each item is the ``data.window_s`` seconds before a "now" second of a
    block, scaled like :class:`WindowDataset`; its label is the stage of
    that second, and it also returns the standardised elapsed time since the
    recording started. What the items are depends on ``mode``:

    - ``train``: one item per ``data.train_stride_s`` seconds of each block.
      Each read picks a random "now" second among the usable ones of that
      slot (the stride point and up to ``train_stride_s - 1`` seconds before
      it), so every second is used over the epochs while an epoch has as many
      windows as fixed windows would.
    - ``pretrain``: as ``train``, also with unlabelled "now" seconds when
      ``data.pretrain_unlabeled`` is set; labels are then -1 and unused.
    - ``val`` / ``test``: every ``val_stride_s`` / ``test_stride_s`` seconds,
      no random choice and no augmentation.

    Parameters
    ----------
    cfg : Config
        Run settings.
    arrays : dict
        Output of :func:`load_dataset` with ``data.rolling``.
    subjects : list of str
        Subjects to include.
    center, scale : np.ndarray
        Per-electrode robust scaling ``[C]`` fitted on the training subjects.
    elapsed_mean, elapsed_std : float
        Standardisation of the elapsed time in hours, from the training windows.
    mode : str
        ``train``, ``pretrain``, ``val`` or ``test``.
    """

    def __init__(self, cfg: Config, arrays: Arrays, subjects: list[str], center: np.ndarray,
                 scale: np.ndarray, elapsed_mean: float, elapsed_std: float, mode: str):
        data = cfg.data
        self.cfg, self.mode = cfg, mode
        self.center, self.scale = center[:, None], scale[:, None]
        self.elapsed_mean, self.elapsed_std = elapsed_mean, elapsed_std
        self.length = int(round(data.window_s))
        self.sfreq = int(data.sfreq)
        self.blocks = [arrays[s][2] for s in subjects]
        self.augment = mode in ("train", "pretrain")
        labelled = not (mode == "pretrain" and data.pretrain_unlabeled)

        candidates = []
        for index, blocks in enumerate(self.blocks):
            block, end = window_ends(cfg, blocks, self.length, labelled)
            candidates.append((np.full(len(block), index), block, end))
        subject = np.concatenate([c[0] for c in candidates])
        block = np.concatenate([c[1] for c in candidates])
        end = np.concatenate([c[2] for c in candidates])
        stage = np.array([self.blocks[s]["stage"][b, e - 1]
                          for s, b, e in zip(subject, block, end)], dtype=np.int64)
        elapsed = np.array([self.blocks[s]["elapsed"][b, e - 1] for s, b, e in
                            zip(subject, block, end)], dtype=np.float64) / 3600.0
        self.all_stage, self.all_elapsed = stage, elapsed

        if mode in ("train", "pretrain"):
            stride = data.train_stride_s
            slot = (end + stride - 1) // stride
            keys = np.stack([subject, block, slot], 1)
            _, first, groups = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            groups = groups.ravel()
            self.slots = [np.flatnonzero(groups == g) for g in range(len(first))]
            self.subject, self.block, self.end = subject, block, end
            self.y = stage[stage >= 0]
        else:
            stride = data.val_stride_s if mode == "val" else data.test_stride_s
            keep = end % stride == 0
            self.subject, self.block, self.end = subject[keep], block[keep], end[keep]
            self.slots = None
            self.y = stage[keep]
            self.elapsed = elapsed[keep]
            within = np.array([inside_one_epoch(self.blocks[s], np.array([b]), np.array([e]),
                                                self.length)[0]
                               for s, b, e in zip(self.subject, self.block, self.end)], dtype=bool)
            self.aligned = (self.end % data.train_stride_s == 0) & within
            self.epoch_id = np.array([s * 1_000_000 + self.blocks[s]["epoch"][b, e - 1]
                                      for s, b, e in zip(self.subject, self.block, self.end)],
                                     dtype=np.int64)

    def __len__(self) -> int:
        """Number of items per epoch.

        Returns
        -------
        int
            Slots for ``train`` / ``pretrain``, windows otherwise.
        """
        return len(self.slots) if self.slots is not None else len(self.end)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One scaled window, its stage and its standardised elapsed time.

        The augmentations of the training items are those of
        :class:`WindowDataset`.

        Parameters
        ----------
        index : int
            Slot (``train`` / ``pretrain``) or window (``val`` / ``test``).

        Returns
        -------
        x : torch.Tensor
            Window ``[C, window samples]``, clipped to ``[-20, 20]``.
        y : torch.Tensor
            Stage index of the last second, -1 for unlabelled pretraining windows.
        elapsed : torch.Tensor
            Standardised hours since the recording started, at the end of the window.
        """
        if self.slots is not None:
            members = self.slots[index]
            index = int(members[int(torch.randint(len(members), (1,)))])
        s, b, e = int(self.subject[index]), int(self.block[index]), int(self.end[index])
        blocks = self.blocks[s]
        start = (e - self.length) * self.sfreq
        window = blocks["X"][b, :, start:e * self.sfreq].astype(np.float32)
        x = torch.from_numpy(np.clip((window - self.center) / self.scale, -20, 20)
                             .astype(np.float32))
        if self.augment:
            augment = self.cfg.augment
            shift = int(torch.randint(-augment.time_roll, augment.time_roll + 1, (1,)))
            x = torch.roll(x, shift, -1)
            x = x * (1.0 + augment.scale * (2 * torch.rand(1) - 1))
            x = x * (torch.rand(x.shape[0], 1) > augment.channel_drop)
        hours = blocks["elapsed"][b, e - 1] / 3600.0
        elapsed = (hours - self.elapsed_mean) / self.elapsed_std
        return (x, torch.tensor(int(blocks["stage"][b, e - 1]), dtype=torch.long),
                torch.tensor(elapsed, dtype=torch.float32))


def build_rolling_loaders(cfg: Config, arrays: Arrays,
                          split: dict[str, list[str]]) -> dict[str, DataLoader]:
    """Loaders of rolling windows; scaling and time statistics from the training subjects.

    Parameters
    ----------
    cfg : Config
        Run settings.
    arrays : dict
        Output of :func:`load_dataset` with ``data.rolling``.
    split : dict
        Subject ids per split.

    Returns
    -------
    dict
        ``train``, ``val`` and ``test`` loaders, and ``pretrain`` when
        ``data.pretrain_unlabeled`` is set.

    Raises
    ------
    ValueError
        If balanced sampling is requested, which rolling windows do not support.
    """
    if cfg.finetune.balanced_sampling:
        raise ValueError("finetune.balanced_sampling is not supported with data.rolling")
    n_channels = next(iter(arrays.values()))[0].shape[1]
    samples = np.concatenate([arrays[s][0][:, :, ::7].transpose(0, 2, 1).reshape(-1, n_channels)
                              for s in split["train"]])
    scaler = RobustScaler().fit(samples)
    center, scale = scaler.center_.astype(np.float32), scaler.scale_.astype(np.float32)

    probe = RollingDataset(cfg, arrays, split["train"], center, scale, 0.0, 1.0, "train")
    hours = probe.all_elapsed[probe.all_stage >= 0]
    elapsed_mean, elapsed_std = float(hours.mean()), float(hours.std() + 1e-6)

    modes = {"train": "train", "val": "val", "test": "test"}
    if cfg.data.pretrain_unlabeled:
        modes["pretrain"] = "pretrain"
    loaders = {}
    for name, mode in modes.items():
        subjects = split["train"] if name == "pretrain" else split[name]
        dataset = RollingDataset(cfg, arrays, subjects, center, scale, elapsed_mean, elapsed_std,
                                 mode)
        shuffle = mode in ("train", "pretrain")
        loaders[name] = DataLoader(dataset, cfg.loader.batch_size, shuffle=shuffle,
                                   drop_last=shuffle, num_workers=cfg.loader.num_workers)
    return loaders


class EpochDataset(Dataset):
    """Scored 30 s epochs of whole nights, alone or in runs (Exp 14 and the final reruns).

    Each epoch is scaled per electrode with the training statistics and
    clipped to ``[-20, 20]``, as :class:`WindowDataset` does, and training
    epochs get its augmentations. Which epochs are items depends on ``mode``:

    - ``train``: scored epochs whose peak is at most ``data.artifact_uv``.
    - ``pretrain``: the same, plus the clean lights-on epochs when
      ``data.pretrain_unlabeled`` is set (label -1, unused by stage 1).
    - ``val`` / ``test``: every scored epoch, artifacts included, so the
      score covers the whole scored night.

    With ``context`` above 1 an item is a run of that many consecutive
    epochs of one night. ``train`` cuts each night into runs of ``context``
    epochs and moves each run back by a random 0 to ``context - 1`` epochs
    whenever it is read, so every epoch is seen at every position; every
    scored, clean epoch of the run is labelled and the rest are -1.
    ``val`` / ``test`` have one run per scored epoch, ending at it (or, near
    the start of the night, starting at the first epoch), labelled only at
    that epoch, so each scored epoch is predicted once from its own past.

    Parameters
    ----------
    cfg : Config
        Run settings.
    arrays : dict
        Output of :func:`load_dataset` with ``data.scored_epochs``.
    subjects : list of str
        Subjects to include.
    center, scale : np.ndarray
        Per-electrode robust scaling ``[C]`` fitted on the training subjects.
    mode : str
        ``train``, ``pretrain``, ``val`` or ``test``.
    context : int
        Epochs per item.

    Raises
    ------
    ValueError
        If ``context`` is above 1 for ``pretrain``, or longer than a night.
    """

    def __init__(self, cfg: Config, arrays: Arrays, subjects: list[str], center: np.ndarray,
                 scale: np.ndarray, mode: str, context: int):
        data = cfg.data
        if context > 1 and mode == "pretrain":
            raise ValueError("stage 1 reads single epochs")
        self.cfg, self.mode, self.context = cfg, mode, context
        self.center, self.scale = center[:, None], scale[:, None]
        self.epochs = [arrays[s][2] for s in subjects]
        self.augment = mode in ("train", "pretrain")

        subject, row, labels, clean_rows = [], [], [], []
        for index, epochs in enumerate(self.epochs):
            stage, clean = epochs["stage"].astype(np.int64), epochs["peak"] <= data.artifact_uv
            if context > len(stage):
                raise ValueError(f"{subjects[index]} has fewer than {context} epochs")
            if mode == "train":
                usable = clean & (stage >= 0)
            elif mode == "pretrain":
                usable = clean & ((stage >= 0) | data.pretrain_unlabeled)
            else:
                usable = stage >= 0
            labels.append(stage[usable] if context == 1 or mode != "train"
                          else stage[clean & (stage >= 0)])
            if context > 1 and mode == "train":
                slots = np.arange(-(-len(stage) // context))
                subject.append(np.full(len(slots), index))
                row.append(slots * context)
                continue
            found = np.flatnonzero(usable)
            subject.append(np.full(len(found), index))
            row.append(found)
            clean_rows.append(clean[found])
        self.subject, self.row = np.concatenate(subject), np.concatenate(row)
        self.y = np.concatenate(labels)
        self.clean = np.concatenate(clean_rows) if clean_rows else None

    def __len__(self) -> int:
        """Number of items per epoch.

        Returns
        -------
        int
            Epochs, or runs of epochs.
        """
        return len(self.row)

    def scaled(self, signal: np.ndarray) -> torch.Tensor:
        """Scale stored microvolts per electrode and clip, as the other datasets do.

        Parameters
        ----------
        signal : np.ndarray
            ``[..., C, samples]`` float16 microvolts.

        Returns
        -------
        torch.Tensor
            float32 of the same shape, in ``[-20, 20]``.
        """
        return torch.from_numpy(np.clip((signal.astype(np.float32) - self.center) / self.scale,
                                        -20, 20).astype(np.float32))

    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        """The augmentations of :class:`WindowDataset`, drawn for one epoch.

        Parameters
        ----------
        x : torch.Tensor
            One epoch ``[C, samples]``.

        Returns
        -------
        torch.Tensor
            The perturbed epoch.
        """
        augment = self.cfg.augment
        shift = int(torch.randint(-augment.time_roll, augment.time_roll + 1, (1,)))
        x = torch.roll(x, shift, -1)
        x = x * (1.0 + augment.scale * (2 * torch.rand(1) - 1))
        return x * (torch.rand(x.shape[0], 1) > augment.channel_drop)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One scaled epoch and its stage, or a run of epochs and their labels.

        Parameters
        ----------
        index : int
            Item index.

        Returns
        -------
        x : torch.Tensor
            ``[C, samples]``, or ``[context, C, samples]``.
        y : torch.Tensor
            The stage (-1 for an unlabelled pretraining epoch), or ``[context]``
            labels with -1 where an epoch is not scored.
        """
        epochs = self.epochs[int(self.subject[index])]
        row, n_epochs = int(self.row[index]), len(epochs["stage"])
        if self.context == 1:
            x = self.scaled(epochs["X"][row])
            if self.augment:
                x = self.perturb(x)
            return x, torch.tensor(int(epochs["stage"][row]), dtype=torch.long)

        stage = epochs["stage"].astype(np.int64)
        if self.mode == "train":
            start = row - int(torch.randint(self.context, (1,)))
            start = min(max(start, 0), n_epochs - self.context)
            run = slice(start, start + self.context)
            labels = np.where(epochs["peak"][run] <= self.cfg.data.artifact_uv, stage[run], -1)
        else:
            start = min(max(row - self.context + 1, 0), n_epochs - self.context)
            run = slice(start, start + self.context)
            labels = np.full(self.context, -1, dtype=np.int64)
            labels[row - start] = stage[row]
        x = self.scaled(epochs["X"][run])
        if self.augment:
            x = torch.stack([self.perturb(epoch) for epoch in x])
        return x, torch.from_numpy(labels.astype(np.int64))


def build_epoch_loaders(cfg: Config, arrays: Arrays,
                        split: dict[str, list[str]]) -> dict[str, DataLoader]:
    """Loaders of scored epochs; the scaler is fitted on the training subjects only.

    Stage 2 and the test use ``finetune.batch_size`` items of
    ``data.context_epochs`` epochs. Stage 1 always reads single epochs in
    batches of ``loader.batch_size``: ``pretrain`` from the training
    subjects, and ``pretrain_val`` from the validation subjects in one fixed
    random order, so a report that reads only its first batches (the
    reconstruction report) still covers every validation subject and stage.

    Parameters
    ----------
    cfg : Config
        Run settings.
    arrays : dict
        Output of :func:`load_dataset` with ``data.scored_epochs``.
    split : dict
        Subject ids per split.

    Runs of epochs also get ``test_single``, the test epochs one by one, for
    the figures that read the backbone of single epochs. With
    ``evaluation.curve_batches`` above 0 there are also
    ``train_eval_pretrain`` (single epochs) and ``train_eval`` (items as in
    stage 2): that many batches of training epochs, drawn once in a fixed
    order from a generator of their own and read exactly as the validation
    epochs are (every scored epoch, no augmentation), for the training
    curves.

    Returns
    -------
    dict
        ``train``, ``val``, ``test``, ``pretrain`` and ``pretrain_val`` loaders,
        and the curve loaders when enabled.
    """
    data, loader = cfg.data, cfg.loader
    if cfg.finetune.balanced_sampling:
        raise ValueError("finetune.balanced_sampling is not supported with data.scored_epochs")
    samples = []
    for subject in split["train"]:
        epochs = arrays[subject][2]
        rows = np.flatnonzero((epochs["stage"] >= 0) & (epochs["peak"] <= data.artifact_uv))
        for first in range(0, len(rows), 64):
            chunk = epochs["X"][rows[first:first + 64]][:, :, ::data.scaler_stride]
            samples.append(chunk.transpose(0, 2, 1).reshape(-1, chunk.shape[1]))
    scaler = RobustScaler().fit(np.concatenate(samples).astype(np.float32))
    center, scale = scaler.center_.astype(np.float32), scaler.scale_.astype(np.float32)

    batch = cfg.finetune.batch_size or loader.batch_size
    loaders = {}
    for name in ("train", "val", "test"):
        dataset = EpochDataset(cfg, arrays, split[name], center, scale, name, data.context_epochs)
        shuffle = name == "train"
        loaders[name] = DataLoader(dataset, batch, shuffle=shuffle, drop_last=shuffle,
                                   num_workers=loader.num_workers)
    pretrain = EpochDataset(cfg, arrays, split["train"], center, scale, "pretrain", 1)
    loaders["pretrain"] = DataLoader(pretrain, loader.batch_size, shuffle=True, drop_last=True,
                                     num_workers=loader.num_workers)
    held_out = EpochDataset(cfg, arrays, split["val"], center, scale, "val", 1)
    order = np.random.default_rng(cfg.seed).permutation(len(held_out)).tolist()
    loaders["pretrain_val"] = DataLoader(held_out, loader.batch_size, sampler=order,
                                         num_workers=loader.num_workers)
    if data.context_epochs > 1:
        single = EpochDataset(cfg, arrays, split["test"], center, scale, "test", 1)
        loaders["test_single"] = DataLoader(single, batch, num_workers=loader.num_workers)
    if cfg.evaluation.curve_batches > 0:
        draw = np.random.default_rng(cfg.seed + 1)
        for name, context, size in (("train_eval_pretrain", 1, loader.batch_size),
                                    ("train_eval", data.context_epochs, batch)):
            seen = EpochDataset(cfg, arrays, split["train"], center, scale, "val", context)
            order = draw.permutation(len(seen))[:cfg.evaluation.curve_batches * size].tolist()
            loaders[name] = DataLoader(seen, size, sampler=order, num_workers=loader.num_workers)
    return loaders


class WindowDataset(Dataset):
    """Windows of several subjects, scaled and optionally perturbed.

    Parameters
    ----------
    cfg : Config
        Run settings (augmentation strengths).
    arrays : dict
        ``subject -> (X, y)``.
    subjects : list of str
        Subjects to include.
    scaler : RobustScaler
        Per-channel scaler fitted on the training subjects.
    augment : bool
        Perturb each window; training split only.
    """

    def __init__(self, cfg: Config, arrays: Arrays, subjects: list[str],
                 scaler: RobustScaler, augment: bool = False):
        self.X = np.concatenate([arrays[s][0] for s in subjects])
        self.y = np.concatenate([arrays[s][1] for s in subjects])
        self.cfg, self.scaler, self.augment = cfg, scaler, augment

    def __len__(self) -> int:
        """Number of windows.

        Returns
        -------
        int
            Windows in the dataset.
        """
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One scaled window and its stage.

        The augmentations keep the label: a circular time shift, an amplitude
        change and dropped electrodes.

        Parameters
        ----------
        index : int
            Position in the dataset.

        Returns
        -------
        x : torch.Tensor
            Window ``[C, samples]``, clipped to ``[-20, 20]``.
        y : torch.Tensor
            Stage index.
        """
        x = self.scaler.transform(self.X[index].T).T.astype(np.float32)
        x = torch.from_numpy(np.clip(x, -20, 20))

        if self.augment:
            augment = self.cfg.augment
            shift = int(torch.randint(-augment.time_roll, augment.time_roll + 1, (1,)))
            x = torch.roll(x, shift, -1)
            x = x * (1.0 + augment.scale * (2 * torch.rand(1) - 1))
            x = x * (torch.rand(x.shape[0], 1) > augment.channel_drop)

        return x, torch.tensor(self.y[index])


def split_subjects(cfg: Config, arrays: Arrays, max_draws: int = 1000) -> dict[str, list[str]]:
    """Divide subjects into train, validation and test, never sharing a subject.

    The draw is repeated until validation and test both contain every stage,
    since a missing stage makes macro-F1 average in an F1 of 0 for it. With
    ``split.n_folds`` set, the test subjects of that draw stay the test set and
    the other subjects are divided into folds instead (:func:`fold_split`).
    With ``split.n_test_groups`` set, every subject is tested once across the
    runs of a rotation (:func:`rotation_split`); the draw above is then not made.

    Parameters
    ----------
    cfg : Config
        Run settings (seed and split sizes).
    arrays : dict
        ``subject -> (X, y)``.
    max_draws : int
        Draws to try before giving up.

    Returns
    -------
    dict
        Sorted subject ids for ``train``, ``val`` and ``test``.

    Raises
    ------
    RuntimeError
        If no draw puts every stage in both validation and test.
    """
    if cfg.split.n_test_groups:
        return rotation_split(cfg, arrays, max_draws)
    n_val, n_test = cfg.split.n_val_subjects, cfg.split.n_test_subjects
    n_stages = len(cfg.data.stages)
    rng = np.random.default_rng(cfg.seed)
    for _ in range(max_draws):
        order = [str(s) for s in rng.permutation(sorted(arrays))]
        split = {"train": sorted(order[n_val + n_test:]), "val": sorted(order[:n_val]),
                 "test": sorted(order[n_val:n_val + n_test])}
        if all(len(np.unique(np.concatenate([arrays[s][1] for s in split[name]]))) == n_stages
               for name in ("val", "test")):
            return fold_split(cfg, arrays, split["test"], max_draws) if cfg.split.n_folds else split
    raise RuntimeError("no split puts every stage in both validation and test")


def rotation_split(cfg: Config, arrays: Arrays, max_draws: int = 1000) -> dict[str, list[str]]:
    """One run of a rotation in which every subject is tested exactly once.

    All subjects are shuffled with the run seed and cut into
    ``split.n_test_groups`` groups of nearly equal size, so the partition is
    the same whichever group a run tests. Group ``split.test_group`` is the
    test set and the next group, cyclically, the validation set, so every
    subject is also used once for validation; the remaining groups train. The
    shuffle is repeated until every group contains every stage, because
    validation and test macro-F1 would otherwise average in a missing stage.
    A fixed test set of a few subjects measures only how much the result
    depends on training and selection; rotating the test set also measures how
    much it depends on the test subjects, which is the larger source of spread.

    Parameters
    ----------
    cfg : Config
        Run settings (seed, ``split.n_test_groups``, ``split.test_group``).
    arrays : dict
        ``subject -> (X, y)``.
    max_draws : int
        Shuffles to try before giving up.

    Returns
    -------
    dict
        Sorted subject ids for ``train``, ``val`` and ``test``.

    Raises
    ------
    ValueError
        If fewer than 3 groups are asked for, ``test_group`` is not one of the
        groups, or ``n_folds`` is also set.
    RuntimeError
        If no shuffle puts every stage in every group.
    """
    n_groups, group = cfg.split.n_test_groups, cfg.split.test_group
    if n_groups < 3 or not 0 <= group < n_groups:
        raise ValueError(f"split.test_group must be in [0, {n_groups}) with n_test_groups >= 3, "
                         f"got group {group} of {n_groups}")
    if cfg.split.n_folds:
        raise ValueError("split.n_folds and split.n_test_groups cannot both be set")
    n_stages = len(cfg.data.stages)
    rng = np.random.default_rng(cfg.seed)
    for _ in range(max_draws):
        groups = np.array_split(rng.permutation(sorted(arrays)), n_groups)
        if all(len(np.unique(np.concatenate([arrays[s][1] for s in members]))) == n_stages
               for members in groups):
            test = sorted(str(s) for s in groups[group])
            val = sorted(str(s) for s in groups[(group + 1) % n_groups])
            return {"train": sorted(set(map(str, arrays)) - set(test) - set(val)), "val": val,
                    "test": test}
    raise RuntimeError(f"no shuffle puts every stage in each of {n_groups} groups")


def fold_split(cfg: Config, arrays: Arrays, test: list[str],
               max_draws: int = 1000) -> dict[str, list[str]]:
    """Cross-validation folds over the subjects outside the fixed test set.

    The subjects that are not test subjects are shuffled with the run seed and
    cut into ``split.n_folds`` folds of nearly equal size; fold
    ``split.fold`` is the validation set and the others are the training set.
    The shuffle is repeated until every fold contains every stage, so the
    partition is the same whichever fold a run uses, and validation macro-F1
    never averages in a missing stage. The test set is the same for every
    fold, so the folds' test scores show how much the result depends on which
    subjects the model is trained and selected on.

    Parameters
    ----------
    cfg : Config
        Run settings (seed, ``split.n_folds``, ``split.fold``).
    arrays : dict
        ``subject -> (X, y)``.
    test : list of str
        Test subjects, kept out of every fold.
    max_draws : int
        Shuffles to try before giving up.

    Returns
    -------
    dict
        Sorted subject ids for ``train``, ``val`` and ``test``.

    Raises
    ------
    ValueError
        If ``n_folds`` is below 2 or ``fold`` is not one of the folds.
    RuntimeError
        If no shuffle puts every stage in every fold.
    """
    n_folds, fold = cfg.split.n_folds, cfg.split.fold
    if n_folds < 2 or not 0 <= fold < n_folds:
        raise ValueError(f"split.fold must be in [0, {n_folds}) with n_folds >= 2, "
                         f"got fold {fold} of {n_folds}")
    n_stages = len(cfg.data.stages)
    development = sorted(set(arrays) - set(test))
    rng = np.random.default_rng(cfg.seed)
    for _ in range(max_draws):
        folds = np.array_split(rng.permutation(development), n_folds)
        if all(len(np.unique(np.concatenate([arrays[s][1] for s in members]))) == n_stages
               for members in folds):
            val = sorted(str(s) for s in folds[fold])
            return {"train": sorted(set(development) - set(val)), "val": val,
                    "test": sorted(test)}
    raise RuntimeError(f"no shuffle puts every stage in each of {n_folds} folds")


def build_loaders(cfg: Config, arrays: Arrays,
                  split: dict[str, list[str]]) -> dict[str, DataLoader]:
    """One loader per split; the scaler is fitted on the training subjects only.

    Parameters
    ----------
    cfg : Config
        Run settings.
    arrays : dict
        ``subject -> (X, y)``.
    split : dict
        Subject ids per split.

    Returns
    -------
    dict
        ``train`` (shuffled, augmented, optionally balanced), ``val`` and ``test`` loaders.
    """
    if cfg.data.scored_epochs:
        return build_epoch_loaders(cfg, arrays, split)
    if cfg.finetune.batch_size:
        raise ValueError("finetune.batch_size is only supported with data.scored_epochs")
    if cfg.data.rolling:
        return build_rolling_loaders(cfg, arrays, split)
    n_channels = next(iter(arrays.values()))[0].shape[1]
    scaler = RobustScaler().fit(
        np.concatenate([arrays[s][0][::7] for s in split["train"]])
        .transpose(0, 2, 1).reshape(-1, n_channels))

    loaders = {}
    for name, subjects in split.items():
        train = name == "train"
        dataset = WindowDataset(cfg, arrays, subjects, scaler, augment=train)
        sampler = None
        if train and cfg.finetune.balanced_sampling:
            weight = 1.0 / np.sqrt(np.bincount(dataset.y, minlength=len(cfg.data.stages)))
            sampler = WeightedRandomSampler(weight[dataset.y], len(dataset), replacement=True)
        loaders[name] = DataLoader(dataset, cfg.loader.batch_size, sampler=sampler,
                                   shuffle=train and sampler is None, drop_last=train,
                                   num_workers=cfg.loader.num_workers)
    return loaders
