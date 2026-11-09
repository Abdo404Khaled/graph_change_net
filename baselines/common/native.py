from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np

import data
from config import Config


def subject_files(cfg: Config) -> dict[str, tuple[Path, Path]]:
    """EDF and hypnogram of every subject, as our pipeline finds them.

    Parameters
    ----------
    cfg : Config
        Reference configuration (``paths.data_root``).

    Returns
    -------
    dict
        ``subject -> (edf_path, hypnogram_path)``.
    """
    return {subject: (edf, hyp) for subject, edf, hyp in data.find_subjects(cfg.paths.data_root)}


def night_epochs(cfg: Config, subject: str) -> dict[str, np.ndarray]:
    """Every hypnogram row of a night that lies inside the recording, as our pipeline keeps it.

    Read from the metadata our pipeline cached, so a native pipeline epochs
    the night at exactly the same onsets and with exactly the same labels.

    Parameters
    ----------
    cfg : Config
        Reference configuration.
    subject : str
        Subject id.

    Returns
    -------
    dict
        ``onset`` (seconds) and ``stage`` (index, -1 for lights on and any
        other unscored row) of every row, in time order, and ``peak``.

    Raises
    ------
    FileNotFoundError
        If our pipeline has not cached this night.
    """
    channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
    signature = data.preprocessing_signature(cfg, channels)
    path = cfg.paths.cache_dir / f"{subject}_{signature}_epochs_meta.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the reference pipeline first")
    meta = np.load(path)
    return {key: meta[key] for key in ("onset", "stage", "peak")}


def read_channels(edf_path: Path, channels: list, transform: Callable[[np.ndarray], np.ndarray]
                  ) -> np.ndarray:
    """Whole-night signals of chosen channels or bipolar pairs, one channel at a time.

    Names are matched as our pipeline matches them (:func:`data.match_channels`,
    which also accepts ``-Ref`` suffixes and case differences). Each entry of
    ``channels`` is a name, or a pair ``[a, b]`` read as ``a - b``. Each
    channel is converted to microvolts and passed through ``transform``
    (for example a resampling) before the next is read, so a night never has
    to be held at 1 kHz for every channel at once.

    Parameters
    ----------
    edf_path : Path
        EDF recording.
    channels : list
        Channel names or ``[a, b]`` pairs.
    transform : callable
        Applied to each 1-D microvolt signal.

    Returns
    -------
    np.ndarray
        ``[n_channels, samples]`` float32.

    Raises
    ------
    KeyError
        If a requested channel is not in the recording.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    wanted = sorted({name.upper() for entry in channels
                     for name in ([entry] if isinstance(entry, str) else entry)})
    raw.rename_channels(data.match_channels(raw.ch_names, wanted))
    present = {name.upper(): name for name in raw.ch_names}

    def signal(name: str) -> np.ndarray:
        """One channel in microvolts.

        Parameters
        ----------
        name : str
            Channel name.

        Returns
        -------
        np.ndarray
            1-D float64 signal.

        Raises
        ------
        KeyError
            If the channel is missing.
        """
        if name.upper() not in present:
            raise KeyError(f"{edf_path.name}: no channel {name}")
        return raw.get_data(picks=[present[name.upper()]])[0] * 1e6

    rows = []
    for entry in channels:
        x = signal(entry) if isinstance(entry, str) else signal(entry[0]) - signal(entry[1])
        rows.append(np.asarray(transform(x), dtype=np.float32))
    raw.close()
    return np.stack(rows)


def cached_night(path: Path, build: Callable[[], np.ndarray]) -> np.ndarray:
    """Load a preprocessed night from its cache, building it once if missing.

    Parameters
    ----------
    path : Path
        ``.npy`` cache file.
    build : callable
        Returns the array when the cache is missing.

    Returns
    -------
    np.ndarray
        The cached array, memory-mapped read-only.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npy")
        np.save(partial, build())
        partial.replace(path)
    return np.load(path, mmap_mode="r")
