"""Hypno-PC epoch features, re-implemented from the authors' code.

Upstream is ``upstream/analysis_code`` at commit ``d3b0eec``; every function
cites the lines it re-implements. The upstream modules are not imported: they
import ``yasa`` and ``picard`` at module level, which this baseline does not
need, and the repository has no license.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy

import data

SPECTRAL_FEATURES = ("root_total_power", "spectral_entropy", "spectral_slope",
                     "spectral_intercept")


def welch(signal: np.ndarray, sfreq: float, band: list[float], n_fft: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectrum with the settings of the upstream ``compute_psd`` calls.

    Upstream calls ``Epochs.compute_psd(method="welch", n_fft=min(4 * sfreq,
    n_samples), fmin, fmax, remove_dc=False)`` (``preprocessing_main.py:108-135,
    155-179``), which runs :func:`mne.time_frequency.psd_array_welch` with its
    defaults: ``n_per_seg = n_fft``, no overlap, Hamming window, mean over
    segments. Upstream epochs have 30 001 samples (``tmax`` is inclusive) and
    ours 30 000; with 4000-sample segments and no overlap both use samples
    0-27 999 only, so the spectra are identical.

    Parameters
    ----------
    signal : np.ndarray
        ``[..., samples]``.
    sfreq : float
        Sampling rate in Hz.
    band : list of float
        ``[fmin, fmax]`` in Hz, both kept.
    n_fft : int
        FFT and segment length in samples.

    Returns
    -------
    psd : np.ndarray
        ``[..., n_frequencies]`` power spectral density.
    freqs : np.ndarray
        Frequencies in Hz.
    """
    return mne.time_frequency.psd_array_welch(
        signal, sfreq, fmin=band[0], fmax=band[1], n_fft=n_fft, n_overlap=0, n_per_seg=None,
        window="hamming", average="mean", remove_dc=False, verbose=False)


def spectral_feature_names(channels: list[str], bands: dict[str, list[float]]) -> list[str]:
    """Names of the 20 EEG features per channel, in the order of :func:`spectral_features`.

    Parameters
    ----------
    channels : list of str
        Electrode names.
    bands : dict
        Band name to ``[fmin, fmax]``.

    Returns
    -------
    list of str
        ``<channel>_<feature>``, channel-major.
    """
    per_channel = list(SPECTRAL_FEATURES)
    for band in bands:
        per_channel += [f"{band}_power", f"log_{band}_power"]
    return [f"{channel}_{feature}" for channel in channels for feature in per_channel]


def spectral_features(psd: np.ndarray, freqs: np.ndarray, bands: dict[str, list[float]]
                      ) -> np.ndarray:
    """The upstream EEG features of every channel of every epoch.

    Re-implements ``extract_spectral_features`` (``preprocessing_utils.py:165-243``),
    vectorised over epochs and channels:

    - ``p = PSD / sum(PSD)`` over the 0.5-40 Hz bins (``:187-188``);
    - ``root_total_power = sqrt(sum(PSD))`` (``:194-199``);
    - ``spectral_entropy = scipy.stats.entropy(p)`` in nats (``:200-205``);
    - ``L = ln(p * e / min(p))``, ``Ln = L / sum(L)``, and an ordinary least
      squares line of ``Ln`` on ``ln f`` gives ``spectral_slope`` and
      ``spectral_intercept`` (``:189-192, 207-223``; the closed form equals
      ``LinearRegression().fit``);
    - per band, ``sum(p)`` and ``sum(Ln)`` over bins with
      ``fmin <= f <= fmax`` (``:225-241``).

    Parameters
    ----------
    psd : np.ndarray
        ``[n_epochs, n_channels, n_frequencies]``.
    freqs : np.ndarray
        Frequencies of the bins.
    bands : dict
        Band name to ``[fmin, fmax]`` (``parameters.yaml:124-132``).

    Returns
    -------
    np.ndarray
        ``[n_epochs, n_channels * 20]`` in the order of :func:`spectral_feature_names`.
    """
    total = psd.sum(axis=-1)
    relative = psd / total[..., None]
    lowest = np.nanmin(relative, axis=-1)
    log_relative = np.log(relative * (np.e / lowest)[..., None])
    normalised_log = log_relative / log_relative.sum(axis=-1)[..., None]
    log_freqs = np.log(freqs)
    centred = log_freqs - log_freqs.mean()
    slope = (normalised_log * centred).sum(axis=-1) / (centred ** 2).sum()
    intercept = normalised_log.mean(axis=-1) - slope * log_freqs.mean()
    columns = [np.sqrt(total), entropy(relative, axis=-1), slope, intercept]
    for low, high in bands.values():
        inside = (freqs >= low) & (freqs <= high)
        columns += [relative[..., inside].sum(axis=-1), normalised_log[..., inside].sum(axis=-1)]
    stacked = np.stack(columns, axis=-1)
    return stacked.reshape(stacked.shape[0], -1)


def ecg_features(ecg: np.ndarray, sfreq: float, max_heart_rate: float) -> np.ndarray:
    """Heart rate and the two mean-normalised RR variabilities of every epoch.

    Re-implements ``extract_ecg_features`` (``preprocessing_utils.py:276-311``):
    ``s = diff(ecg) ** 2``; R peaks from ``find_peaks(s, height=mean(s),
    distance=int(60 / max_heart_rate * sfreq))``; ``RR = diff(peaks) / sfreq``;
    heart rate ``60 / mean(RR)``; SDNN ``std(RR / mean(RR))``; RMSSD
    ``sqrt(mean(diff(RR / mean(RR)) ** 2))``. An epoch with fewer than two
    peaks (three for RMSSD) gives NaN, as upstream does, and is interpolated
    later.

    Parameters
    ----------
    ecg : np.ndarray
        ``[n_epochs, samples]`` bipolar ECG.
    sfreq : float
        Sampling rate in Hz.
    max_heart_rate : float
        Heart rate in beats per minute that sets the refractory distance.

    Returns
    -------
    np.ndarray
        ``[n_epochs, 3]``: heart rate, SDNN, RMSSD.
    """
    distance = int(60 / max_heart_rate * sfreq)
    result = np.full((len(ecg), 3), np.nan)
    for index, signal in enumerate(ecg):
        squared = np.diff(signal) ** 2
        peaks, _ = find_peaks(squared, height=(np.mean(squared), None), distance=distance)
        intervals = np.diff(peaks) / sfreq
        if intervals.size == 0:
            continue
        normalised = intervals / intervals.mean()
        result[index, 0] = 60 / intervals.mean()
        result[index, 1] = normalised.std()
        if intervals.size > 1:
            result[index, 2] = np.sqrt(np.mean(np.diff(normalised) ** 2))
    return result


def native_feature_names(channels: list[str], spec: dict) -> list[str]:
    """Names of all 1669 native features, in the order of :func:`native_night`.

    Parameters
    ----------
    channels : list of str
        EEG electrode names.
    spec : dict
        Native configuration.

    Returns
    -------
    list of str
        EEG, then EMG, EOG and ECG feature names.
    """
    signals = spec["signals"]
    return (spectral_feature_names(channels, spec["bands"]) + list(signals["emg"]["pairs"])
            + list(signals["eog"]["channels"]) + list(signals["eog"]["differences"])
            + list(signals["ecg"]["names"]))


def condition_native(block: np.ndarray, row: dict[str, int], channels: list[str],
                     sfreq: float, signals: dict) -> dict[str, np.ndarray]:
    """Filter and re-reference one stretch of a night as upstream does.

    Re-implements ``preprocess_eeg`` (``preprocessing_utils.py:117-140``) with
    :func:`mne.filter.filter_data`, the function ``Raw.filter`` calls, with the
    same defaults (FIR, firwin, Hamming, zero phase, automatic transition
    bands): high-pass EEG and ECG, EOG and EMG; average reference over the
    EEG; bipolar EMG and ECG, and the EOG difference; then low-pass each
    modality. All steps are linear and time-invariant, so filtering a padded
    stretch gives the whole-night result away from the recording edges.

    Parameters
    ----------
    block : np.ndarray
        ``[n_read_channels, samples]`` in volts.
    row : dict
        Channel name to row of ``block``.
    channels : list of str
        EEG electrode names.
    sfreq : float
        Sampling rate in Hz.
    signals : dict
        ``signals`` section of the native configuration.

    Returns
    -------
    dict
        ``eeg``, ``emg``, ``eog`` and ``ecg`` arrays ``[n, samples]``.
    """
    def band(x: np.ndarray, settings: dict) -> np.ndarray:
        """High-pass then low-pass, as two separate filters.

        Parameters
        ----------
        x : np.ndarray
            ``[n, samples]``.
        settings : dict
            ``highpass`` and ``lowpass`` in Hz.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        x = mne.filter.filter_data(x, sfreq, settings["highpass"], None, verbose=False)
        return mne.filter.filter_data(x, sfreq, None, settings["lowpass"], verbose=False)

    def rows(names: list[str]) -> np.ndarray:
        """Rows of ``block`` for the given channels.

        Parameters
        ----------
        names : list of str
            Channel names.

        Returns
        -------
        np.ndarray
            ``[len(names), samples]``.
        """
        return block[[row[name] for name in names]]

    eeg_settings = signals["eeg"]
    eeg = mne.filter.filter_data(rows(channels), sfreq, eeg_settings["highpass"], None,
                                 verbose=False)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)
    eeg = mne.filter.filter_data(eeg, sfreq, None, eeg_settings["lowpass"], verbose=False)

    eog_settings = signals["eog"]
    eog_names = sorted({name for name in eog_settings["channels"].values()}
                       | {name for pair in eog_settings["differences"].values() for name in pair})
    eog_high = mne.filter.filter_data(rows(eog_names), sfreq, eog_settings["highpass"], None,
                                      verbose=False)
    eog_row = {name: index for index, name in enumerate(eog_names)}
    eog = [eog_high[eog_row[name]] for name in eog_settings["channels"].values()]
    eog += [eog_high[eog_row[a]] - eog_high[eog_row[b]]
            for a, b in eog_settings["differences"].values()]
    eog = mne.filter.filter_data(np.stack(eog), sfreq, None, eog_settings["lowpass"],
                                 verbose=False)

    emg_settings = signals["emg"]
    emg = np.stack([rows([a])[0] - rows([b])[0] for a, b in emg_settings["pairs"].values()])
    emg = band(emg, emg_settings)

    ecg_settings = signals["ecg"]
    anode, cathode = ecg_settings["pair"]
    ecg = band(rows([anode])[0:1] - rows([cathode])[0:1], ecg_settings)
    return {"eeg": eeg, "emg": emg, "eog": eog, "ecg": ecg}


def native_read_names(channels: list[str], signals: dict) -> list[str]:
    """Every EDF channel the native pipeline reads, EEG first.

    Parameters
    ----------
    channels : list of str
        EEG electrode names.
    signals : dict
        ``signals`` section of the native configuration.

    Returns
    -------
    list of str
        Upper-case channel names.
    """
    others = list(signals["eog"]["channels"].values())
    others += [name for pair in signals["eog"]["differences"].values() for name in pair]
    others += [name for pair in signals["emg"]["pairs"].values() for name in pair]
    others += list(signals["ecg"]["pair"])
    unique = list(dict.fromkeys(name.upper() for name in others))
    return [name.upper() for name in channels] + unique


def native_night(edf_path: str, onsets: np.ndarray, channels: list[str], spec: dict
                 ) -> tuple[np.ndarray, float]:
    """All native features of the given epochs of one night, read straight from the EDF.

    The night is read in stretches of ``signals.block_epochs`` epochs with
    ``signals.pad_s`` seconds of real signal on both sides, conditioned by
    :func:`condition_native` and cut into 30 s epochs at the hypnogram onsets
    (upstream epochs from the EDF start every 30 s and joins on time,
    ``preprocessing_main.py:200-234``; the ANPHY onsets are multiples of 30 s,
    so the epochs are the same). Features per epoch: :func:`spectral_features`
    of the EEG, the summed 10-50 Hz EMG spectrum (``preprocessing_utils.py:314-338``),
    the summed 0.3-2 Hz EOG spectrum (``:246-273``) and :func:`ecg_features`.

    Parameters
    ----------
    edf_path : str
        EDF recording.
    onsets : np.ndarray
        Epoch onsets in seconds.
    channels : list of str
        EEG electrode names.
    spec : dict
        Native configuration.

    Returns
    -------
    features : np.ndarray
        ``[n_epochs, n_features]`` in the order of :func:`native_feature_names`.
    seconds : float
        Wall time of the extraction.

    Raises
    ------
    KeyError
        If a channel is missing from the recording.
    """
    started = time.time()
    signals, welch_settings = spec["signals"], spec["welch"]
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    names = native_read_names(channels, signals)
    raw.rename_channels(data.match_channels(raw.ch_names, names))
    present = {name.upper(): name for name in raw.ch_names}
    missing = [name for name in names if name not in present]
    if missing:
        raise KeyError(f"{Path(edf_path).name}: missing channels {missing}")
    picks = [present[name] for name in names]
    row = {name: index for index, name in enumerate(names)}
    sfreq = float(raw.info["sfreq"])
    epoch_samples = int(round(signals["window_s"] * sfreq))
    pad_samples = int(round(signals["pad_s"] * sfreq))
    starts = np.round(np.asarray(onsets) * sfreq).astype(np.int64)
    upper = [name.upper() for name in channels]
    rows_out = []
    for first in range(0, len(starts), signals["block_epochs"]):
        block_starts = starts[first:first + signals["block_epochs"]]
        low = max(0, int(block_starts[0]) - pad_samples)
        high = min(raw.n_times, int(block_starts[-1]) + epoch_samples + pad_samples)
        block = raw.get_data(picks=picks, start=low, stop=high)
        conditioned = condition_native(block, row, upper, sfreq, signals)
        cut = {key: np.stack([value[:, s - low:s - low + epoch_samples] for s in block_starts])
               for key, value in conditioned.items()}
        psd, freqs = welch(cut["eeg"], sfreq, welch_settings["band"], welch_settings["n_fft"])
        emg_psd, _ = welch(cut["emg"], sfreq, signals["emg"]["band"], welch_settings["n_fft"])
        eog_psd, _ = welch(cut["eog"], sfreq, signals["eog"]["band"], welch_settings["n_fft"])
        rows_out.append(np.concatenate([
            spectral_features(psd, freqs, spec["bands"]), emg_psd.sum(axis=-1),
            eog_psd.sum(axis=-1),
            ecg_features(cut["ecg"][:, 0], sfreq, signals["ecg"]["max_heart_rate"])], axis=1))
    raw.close()
    return np.concatenate(rows_out), time.time() - started


def ours_night(signal_path: str, n_epochs: int, center: np.ndarray, scale: np.ndarray,
               sfreq: float, spec: dict) -> tuple[np.ndarray, float]:
    """EEG features of the first epochs of one night, from exactly the tensors our model reads.

    The cached epochs of :func:`data.load_subject_epochs` (1 kHz; notch,
    0.5-45 Hz band-pass, common average) are scaled and clipped by
    :meth:`data.EpochDataset.scaled` with the training subjects' robust
    statistics, then go through :func:`welch` and :func:`spectral_features`.

    Parameters
    ----------
    signal_path : str
        Cached ``[n_epochs, C, samples]`` epochs.
    n_epochs : int
        Number of epochs to process from the start of the night.
    center, scale : np.ndarray
        Per-electrode robust statistics ``[C, 1]`` of the training subjects.
    sfreq : float
        Sampling rate in Hz.
    spec : dict
        Our-configuration settings.

    Returns
    -------
    features : np.ndarray
        ``[n_epochs, C * 20]``.
    seconds : float
        Wall time of the extraction.
    """
    started = time.time()
    epochs = np.load(signal_path, mmap_mode="r")
    scaler = SimpleNamespace(center=center, scale=scale)
    block_epochs, welch_settings = spec["block_epochs"], spec["welch"]
    rows_out = []
    for first in range(0, n_epochs, block_epochs):
        last = min(first + block_epochs, n_epochs)
        scaled = data.EpochDataset.scaled(scaler, np.asarray(epochs[first:last]))
        psd, freqs = welch(scaled.numpy().astype(np.float64), sfreq, welch_settings["band"],
                           welch_settings["n_fft"])
        rows_out.append(spectral_features(psd, freqs, spec["bands"]))
    return np.concatenate(rows_out), time.time() - started
