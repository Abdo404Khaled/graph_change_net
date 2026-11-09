"""Exp 15 fold 0 (trained on 0.5-45 Hz + 60 Hz notch) scored on the test nights under other input filters.

The epochs are cut exactly as data.load_subject_epochs cuts them (same blocks, padding, common
average, float16); only the filter step differs. The scaler stays the one fitted on the
(filtered) training nights, as in training.
"""
import json
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
from config import Config, build, read_yaml  # noqa: E402
from evaluate import predict_proba, scores  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUN = REPO / "runs/exp15_final_fold0"
cfg = build(Config, read_yaml(RUN / "config.yaml"))
split = json.loads((RUN / "metrics.json").read_text())["split"]
HARMONICS = [60.0 * k for k in range(1, 8)]


def f_raw(x, sf):
    return x


def f_no_lowpass(x, sf):
    x = mne.filter.notch_filter(x, sf, HARMONICS, verbose=False)
    return mne.filter.filter_data(x, sf, 0.5, None, verbose=False)


def make_lowpass(hi):
    def f(x, sf):
        x = mne.filter.notch_filter(x, sf, [60.0], verbose=False)
        return mne.filter.filter_data(x, sf, 0.5, hi, verbose=False)
    return f


VARIANTS = {"raw (no band-pass, no notch)": f_raw,
            "0.5 Hz high-pass + notches, no low-pass": f_no_lowpass,
            "0.5-30 Hz": make_lowpass(30.0), "0.5-20 Hz": make_lowpass(20.0)}


def build_variants(subject, edf, hyp, channels, meta):
    paths = {name: OUT / f"{subject}_{i}.npy" for i, name in enumerate(VARIANTS)}
    if all(p.exists() for p in paths.values()):
        return {n: np.load(p, mmap_mode="r") for n, p in paths.items()}
    d = cfg.data
    raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    raw.rename_channels(data.match_channels(raw.ch_names, channels))
    keep = [c for c in channels if c in raw.ch_names]
    raw.pick(keep)
    sf = float(raw.info["sfreq"])
    epoch_n, pad_n = int(round(d.window_s * sf)), int(d.pad_s * sf)
    starts = np.round(meta["onset"] * sf).astype(np.int64)
    X = {n: np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(len(starts), len(keep), epoch_n))
         for n, p in paths.items()}
    rows = max(1, int(d.block_s // d.window_s))
    for first in range(0, len(starts), rows):
        last = min(first + rows, len(starts))
        begin, end = int(starts[first]), int(starts[last - 1]) + epoch_n
        lo, hi = max(0, begin - pad_n), min(raw.n_times, end + pad_n)
        signal = raw.get_data(start=lo, stop=hi) * 1e6
        for name, f in VARIANTS.items():
            s = f(signal.copy(), sf)
            s = s - s.mean(axis=0, keepdims=True)
            ep = s[:, begin - lo:end - lo].reshape(len(keep), last - first, epoch_n).transpose(1, 0, 2)
            X[name][first:last] = ep.astype(np.float32)
    raw.close()
    for a in X.values():
        a.flush()
    return {n: np.load(p, mmap_mode="r") for n, p in paths.items()}


arrays, channels, coords = data.load_dataset(cfg)
loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)
model = GraphChangeNet(cfg, len(channels), coords).to(cfg.device)
load_weights(model, RUN / "classify_best.pt", cfg.device)
test = loaders["test"]
results = {}
true, prob = predict_proba(cfg, model, test)
results["our pipeline (0.5-45 Hz + 60 Hz notch)"] = scores(cfg, true, prob.argmax(1))
print("reference", results["our pipeline (0.5-45 Hz + 60 Hz notch)"], flush=True)
files = data.find_subjects(cfg.paths.data_root)
paths = {s: (e, h) for s, e, h in files}
original = [ep["X"] for ep in test.dataset.epochs]
built = {s: build_variants(s, *paths[s], channels, arrays[s][2]) for s in split["test"]}
for name in VARIANTS:
    for i, s in enumerate(split["test"]):
        test.dataset.epochs[i]["X"] = built[s][name]
    true, prob = predict_proba(cfg, model, test)
    results[name] = scores(cfg, true, prob.argmax(1))
    print(name, results[name], flush=True)
for i, x in enumerate(original):
    test.dataset.epochs[i]["X"] = x
(OUT / "input_variants.json").write_text(json.dumps(results, indent=1))
