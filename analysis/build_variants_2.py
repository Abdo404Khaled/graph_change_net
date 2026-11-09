"""Two more input variants of the 5 fold-0 test nights, cut exactly as input_variants.py cuts them.

4: notches at 60-420 Hz only (no high-pass, no low-pass): is it drift that breaks raw input?
5: 0.5 Hz high-pass only (no notch, no low-pass): is it line noise?
Raw (0) has neither, the no-low-pass variant (1) has both. Saved as <night>_4.npy / _5.npy, float32.
"""
import json
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
from config import load_run_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
RUN = REPO / "runs/exp15_final_fold0"
cfg = load_run_config(RUN, {})
split = json.loads((RUN / "metrics.json").read_text())["split"]
HARMONICS = [60.0 * k for k in range(1, 8)]
VARIANTS = {4: lambda x, sf: mne.filter.notch_filter(x, sf, HARMONICS, verbose=False),
            5: lambda x, sf: mne.filter.filter_data(x, sf, 0.5, None, verbose=False)}
channels, _ = data.read_montage(data.find_montage(cfg.paths.data_root))
files = {s: (e, h) for s, e, h in data.find_subjects(cfg.paths.data_root)}
d = cfg.data
for subject in split["test"]:
    edf, hyp = files[subject]
    signature = data.preprocessing_signature(cfg, channels)
    meta = np.load(REPO / cfg.paths.cache_dir / f"{subject}_{signature}_epochs_meta.npz")
    raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    raw.rename_channels(data.match_channels(raw.ch_names, channels))
    keep = [c for c in channels if c in raw.ch_names]
    raw.pick(keep)
    sf = float(raw.info["sfreq"])
    epoch_n, pad_n = int(round(d.window_s * sf)), int(d.pad_s * sf)
    starts = np.round(meta["onset"] * sf).astype(np.int64)
    X = {i: np.lib.format.open_memmap(OUT / f"{subject}_{i}.npy", mode="w+", dtype=np.float32,
                                      shape=(len(starts), len(keep), epoch_n)) for i in VARIANTS}
    rows = max(1, int(d.block_s // d.window_s))
    for first in range(0, len(starts), rows):
        last = min(first + rows, len(starts))
        begin, end = int(starts[first]), int(starts[last - 1]) + epoch_n
        lo, hi = max(0, begin - pad_n), min(raw.n_times, end + pad_n)
        signal = raw.get_data(start=lo, stop=hi) * 1e6
        for i, f in VARIANTS.items():
            s = f(signal.copy(), sf)
            s = s - s.mean(axis=0, keepdims=True)
            ep = s[:, begin - lo:end - lo].reshape(len(keep), last - first, epoch_n).transpose(1, 0, 2)
            X[i][first:last] = ep.astype(np.float32)
    raw.close()
    for a in X.values():
        a.flush()
    print(subject, "done", flush=True)
print("all done", flush=True)
