"""Event-locked change of the electrode graph at slow waves and spindles.

For every scored test epoch of fold 0 the trained Exp 15 graph learner gives, per 1 s segment, the
new change ``δ_t`` (``ResidualChange.change``, before accumulation) and the accumulated ``Δ_t``.
Slow waves and spindles are detected on the raw signal (µV, 0.5–45 Hz, average reference):

- slow waves (Massimini et al. 2004): on the mean of the frontal electrodes, band-passed 0.5–4 Hz, a
  negative zero crossing followed by a positive one 0.3–1.0 s later, with a negative peak below the
  amplitude threshold and a peak-to-peak amplitude above the second threshold. ``absolute`` uses
  Massimini's −80 µV / 140 µV (set on a mastoid reference); ``relative`` keeps, per night, the
  candidates whose negative peak is in the lowest quartile of all N2 / N3 candidates of that night,
  because an average reference lowers frontal amplitudes. A second holds a slow wave when the
  negative peak falls in it.
- spindles: on the mean of the central electrodes, band-passed 11–16 Hz, RMS over 0.3 s; above the
  night's mean + 1.5 SD of the N2 RMS for 0.5–3 s. A second holds a spindle when at least 0.5 s of
  it lies inside one.

Tests, per stage (N1, N2, N3; W and R reported for the event rates):

- P1: ``‖δ_t‖`` in event seconds against non-event seconds of the same stage, raw and controlled for
  amplitude (OLS with night fixed effects on log broadband, delta and sigma power of the second);
  CI from a two-stage bootstrap (nights, then epochs);
- P2: the mean ``δ_t`` of event seconds minus that of non-event seconds of the same stage,
  averaged over pairs of scalp regions, with bootstrap CIs;
- P3: slow-wave rate per minute and mean ``‖δ_t‖`` per stage, and the share of seconds above the
  90th percentile of ``‖δ_t‖``.

argv: output JSON path, bootstrap draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

RUN = REPO / "runs/exp15_final_fold0"
STAGES = ("W", "N1", "N2", "N3", "R")
REGIONS = ("front L", "front R", "centre L", "centre R", "back L", "back R", "low L", "low R")


def regions(coords: np.ndarray) -> np.ndarray:
    """Scalp region of every electrode, as in the stage figure (Fig. 10).

    Parameters
    ----------
    coords : np.ndarray
        ``[C, 3]`` positions, x forward, y left, z up.

    Returns
    -------
    np.ndarray
        Region index per electrode (see ``REGIONS``).
    """
    centred = coords - coords.mean(0)
    radius = np.linalg.norm(centred, axis=1)
    elevation = np.degrees(np.arcsin(centred[:, 2] / radius))
    inferior = elevation < np.percentile(elevation, 22)
    front = centred[:, 0] / radius
    band = np.where(inferior, 3, np.where(front > 0.35, 0, np.where(front < -0.35, 2, 1)))
    return band * 2 + (centred[:, 1] < 0).astype(int)


def slow_wave_candidates(signal: np.ndarray, rate: float) -> list[tuple[int, float, float]]:
    """Negative half-waves with zero crossings 0.3–1.0 s apart.

    Parameters
    ----------
    signal : np.ndarray
        Band-passed signal of one epoch, µV.
    rate : float
        Sampling rate, Hz.

    Returns
    -------
    list of tuple
        ``(sample of the negative peak, negative peak µV, peak-to-peak µV)``.
    """
    sign = np.signbit(signal)
    down = np.flatnonzero(~sign[:-1] & sign[1:])
    up = np.flatnonzero(sign[:-1] & ~sign[1:])
    out = []
    for start in down:
        later = up[up > start]
        if not len(later):
            break
        end = later[0]
        if not 0.3 * rate <= end - start <= 1.0 * rate:
            continue
        trough = start + int(np.argmin(signal[start:end + 1]))
        following = signal[end:min(len(signal), end + int(rate))]
        peak = following.max() if len(following) else 0.0
        out.append((trough, float(signal[trough]), float(peak - signal[trough])))
    return out


def spindle_seconds(signal: np.ndarray, rate: float, threshold: float) -> np.ndarray:
    """Seconds of one epoch that are at least half inside a spindle.

    Parameters
    ----------
    signal : np.ndarray
        11–16 Hz band-passed signal, µV.
    rate : float
        Sampling rate, Hz.
    threshold : float
        RMS threshold of the night.

    Returns
    -------
    np.ndarray
        Boolean per second.
    """
    window = int(0.3 * rate)
    power = np.convolve(signal ** 2, np.ones(window) / window, mode="same")
    above = np.sqrt(power) > threshold
    inside = np.zeros_like(above)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    for start, end in zip(edges[::2], edges[1::2]):
        if 0.5 * rate <= end - start <= 3.0 * rate:
            inside[start:end] = True
    return inside.reshape(-1, int(rate)).mean(1) >= 0.5


def bootstrap_index(nights: np.ndarray, draws: int, rng: np.random.Generator) -> list:
    """Two-stage bootstrap draws over nights, then items within each drawn night.

    Parameters
    ----------
    nights : np.ndarray
        Night of each item.
    draws : int
        Number of draws.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    list of np.ndarray
        Item indices per draw.
    """
    groups = [np.flatnonzero(nights == n) for n in np.unique(nights)]
    out = []
    for _ in range(draws):
        chosen = rng.integers(len(groups), size=len(groups))
        out.append(np.concatenate([rng.choice(groups[g], size=len(groups[g])) for g in chosen]))
    return out


def ols_event_effect(y: np.ndarray, event: np.ndarray, covariates: np.ndarray,
                     nights: np.ndarray) -> float:
    """Coefficient of the event indicator in an OLS with covariates and night fixed effects.

    Parameters
    ----------
    y : np.ndarray
        ``‖δ_t‖`` per second.
    event : np.ndarray
        0/1 event indicator.
    covariates : np.ndarray
        ``[n, k]`` amplitude covariates.
    nights : np.ndarray
        Night per second.

    Returns
    -------
    float
        Event coefficient.
    """
    dummies = (nights[:, None] == np.unique(nights)[None]).astype(float)
    design = np.column_stack([event.astype(float), covariates, dummies])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0])


def main():
    """Detect events, compute the per-second change and write every test to one JSON."""
    out_path, draws = Path(sys.argv[1]), int(sys.argv[2])
    cfg = load_run_config(RUN, {})
    split = json.loads((RUN / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(cfg)
    loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)
    test = loaders["test_single"]
    dataset = test.dataset
    rate = float(cfg.data.sfreq)
    region = regions(coords)
    frontal = np.flatnonzero(region <= 1)
    central = np.flatnonzero((region == 2) | (region == 3))
    slow_band = butter(2, [0.5, 4.0], btype="band", fs=rate, output="sos")
    sigma_band = butter(4, [11.0, 16.0], btype="band", fs=rate, output="sos")
    delta_band = slow_band

    model = GraphChangeNet(cfg, len(channels), coords).to(cfg.device)
    load_weights(model, RUN / "classify_best.pt", cfg.device)
    model.eval()
    learner = model.graph_learner
    pair_index = [(a, b) for a in range(len(REGIONS)) for b in range(a, len(REGIONS))]
    masks = []
    for a, b in pair_index:
        m = np.zeros((len(channels), len(channels)), dtype=bool)
        m[np.ix_(region == a, region == b)] = True
        m[np.ix_(region == b, region == a)] = True
        np.fill_diagonal(m, False)
        masks.append(torch.as_tensor(m, device=cfg.device))

    records = {key: [] for key in ("night", "stage", "second", "new_norm", "accumulated_norm",
                                   "log_power", "log_delta", "log_sigma", "frontal_trough",
                                   "frontal_p2p", "sigma_rms_mean")}
    region_means, sigma_signals, candidates = [], [], []
    with torch.no_grad():
        for index, (x, y) in enumerate(test):
            items = range(index * test.batch_size, index * test.batch_size + x.shape[0])
            cells = model.segments(x.to(cfg.device))
            stable = learner.stable_graph()
            new = learner.change(cells, stable)
            accumulated = learner.accumulate(new)
            new_norm = new.flatten(-2).norm(dim=-1).cpu().numpy()
            acc_norm = accumulated.flatten(-2).norm(dim=-1).cpu().numpy()
            region_means.append(torch.stack([(new * m).sum((-1, -2)) / m.sum() for m in masks], -1)
                                .cpu().numpy().astype(np.float32))
            for row, item in enumerate(items):
                night = split["test"][int(dataset.subject[item])]
                epochs = arrays[night][2]
                raw = np.asarray(epochs["X"][int(dataset.row[item])], dtype=np.float32)
                stage = int(epochs["stage"][int(dataset.row[item])])
                slow = sosfiltfilt(slow_band, raw[frontal].mean(0))
                sigma = sosfiltfilt(sigma_band, raw[central].mean(0))
                per_second = raw.reshape(raw.shape[0], 30, -1)
                power = np.log((per_second ** 2).mean((0, 2)) + 1e-6)
                delta = sosfiltfilt(delta_band, raw.mean(0)).reshape(30, -1)
                sig = sigma.reshape(30, -1)
                found = slow_wave_candidates(slow, rate)
                candidates.append((night, stage, found))
                sigma_signals.append(sigma.astype(np.float32))
                for second in range(30):
                    records["night"].append(night)
                    records["stage"].append(stage)
                    records["second"].append(second)
                    records["new_norm"].append(float(new_norm[row, second]))
                    records["accumulated_norm"].append(float(acc_norm[row, second]))
                    records["log_power"].append(float(power[second]))
                    records["log_delta"].append(float(np.log((delta[second] ** 2).mean() + 1e-6)))
                    records["log_sigma"].append(float(np.log((sig[second] ** 2).mean() + 1e-6)))
            print(f"batch {index + 1}/{len(test)}", flush=True) if index % 40 == 0 else None
    table = {k: np.asarray(v) for k, v in records.items() if v}
    region_means = np.concatenate(region_means).reshape(-1, len(pair_index))
    n_epochs = len(candidates)
    nights = table["night"]

    thresholds_relative = {}
    for night in np.unique(nights):
        troughs = [t for n, s, found in candidates if n == night and s in (2, 3) for _, t, _ in found]
        thresholds_relative[night] = float(np.percentile(troughs, 25)) if troughs else -np.inf
    slow_absolute = np.zeros(n_epochs * 30, dtype=bool)
    slow_relative = np.zeros(n_epochs * 30, dtype=bool)
    for e, (night, stage, found) in enumerate(candidates):
        for sample, trough, p2p in found:
            second = min(29, sample // int(rate))
            if trough < -80 and p2p > 140:
                slow_absolute[e * 30 + second] = True
            if trough <= thresholds_relative[night]:
                slow_relative[e * 30 + second] = True
    spindle = np.zeros(n_epochs * 30, dtype=bool)
    for night in np.unique(nights):
        n2 = [e for e, (n, s, _) in enumerate(candidates) if n == night and s == 2]
        if not n2:
            continue
        window = int(0.3 * rate)
        rms = np.concatenate([np.sqrt(np.convolve(sigma_signals[e] ** 2, np.ones(window) / window, "same"))
                              for e in n2])
        threshold = rms.mean() + 1.5 * rms.std()
        for e, (n, s, _) in enumerate(candidates):
            if n == night:
                spindle[e * 30:(e + 1) * 30] = spindle_seconds(sigma_signals[e], rate, threshold)

    rng = np.random.default_rng(0)
    stage, second = table["stage"], table["second"]
    covariates = np.column_stack([table["log_power"], table["log_delta"], table["log_sigma"]])
    result = {"epochs": n_epochs, "nights": sorted(np.unique(nights).tolist()), "draws": draws,
              "relative_trough_threshold_uV": thresholds_relative, "rates_per_minute": {},
              "change_by_stage": {}, "p1": {}, "p2": {}, "regions": list(REGIONS),
              "region_pairs": [f"{REGIONS[a]}–{REGIONS[b]}" for a, b in pair_index]}
    for k, name in enumerate(STAGES):
        here = stage == k
        minutes = here.sum() / 60
        result["rates_per_minute"][name] = {
            "slow_waves_absolute": float(slow_absolute[here].sum() / minutes) if minutes else None,
            "slow_waves_relative": float(slow_relative[here].sum() / minutes) if minutes else None,
            "spindle_seconds_share": float(spindle[here].mean()) if here.any() else None}
        top = np.percentile(table["new_norm"], 90)
        result["change_by_stage"][name] = {"mean_new_norm": float(table["new_norm"][here].mean()),
                                           "share_above_p90": float((table["new_norm"][here] > top).mean()),
                                           "seconds": int(here.sum())}
    for label, event in (("slow_wave_absolute", slow_absolute), ("slow_wave_relative", slow_relative),
                         ("spindle", spindle)):
        for k in (1, 2, 3):
            name = STAGES[k]
            keep = (stage == k) & (second > 0)
            if event[keep].sum() < 20:
                result["p1"][f"{label}_{name}"] = {"event_seconds": int(event[keep].sum())}
                continue
            y, ev, cov, nt = table["new_norm"][keep], event[keep], covariates[keep], nights[keep]
            raw_difference = float(y[ev].mean() - y[~ev].mean())
            controlled = ols_event_effect(y, ev, cov, nt)
            boot_raw, boot_ctrl = [], []
            for idx in bootstrap_index(nt, draws, rng):
                if ev[idx].any() and (~ev[idx]).any():
                    boot_raw.append(y[idx][ev[idx]].mean() - y[idx][~ev[idx]].mean())
                    boot_ctrl.append(ols_event_effect(y[idx], ev[idx], cov[idx], nt[idx]))
            per_night = {n: float(y[(nt == n) & ev].mean() - y[(nt == n) & ~ev].mean())
                         for n in np.unique(nt) if ((nt == n) & ev).any()}
            result["p1"][f"{label}_{name}"] = {
                "event_seconds": int(ev.sum()), "other_seconds": int((~ev).sum()),
                "mean_event": float(y[ev].mean()), "mean_other": float(y[~ev].mean()),
                "raw_difference": raw_difference,
                "raw_ci": [float(np.percentile(boot_raw, 2.5)), float(np.percentile(boot_raw, 97.5))],
                "controlled_difference": controlled,
                "controlled_ci": [float(np.percentile(boot_ctrl, 2.5)), float(np.percentile(boot_ctrl, 97.5))],
                "nights_positive": int(sum(v > 0 for v in per_night.values())), "per_night": per_night}
            pattern = region_means[keep][ev].mean(0) - region_means[keep][~ev].mean(0)
            boots = []
            for idx in bootstrap_index(nt, max(200, draws // 5), rng):
                sub, sev = region_means[keep][idx], ev[idx]
                if sev.any() and (~sev).any():
                    boots.append(sub[sev].mean(0) - sub[~sev].mean(0))
            boots = np.array(boots)
            order = np.argsort(-np.abs(pattern))[:6]
            result["p2"][f"{label}_{name}"] = {
                "top_region_pairs": [{"pair": result["region_pairs"][i], "difference": float(pattern[i]),
                                      "ci": [float(np.percentile(boots[:, i], 2.5)),
                                             float(np.percentile(boots[:, i], 97.5))]} for i in order],
                "all": {result["region_pairs"][i]: float(pattern[i]) for i in range(len(pattern))}}
    out_path.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"rates": result["rates_per_minute"], "change": result["change_by_stage"]}, indent=1))
    for key, value in result["p1"].items():
        if "raw_difference" in value:
            print(f"{key:28s} n={value['event_seconds']:5d}  raw {value['raw_difference']:+.3f} {np.round(value['raw_ci'], 3)}  "
                  f"controlled {value['controlled_difference']:+.3f} {np.round(value['controlled_ci'], 3)}  nights+ {value['nights_positive']}/5")
        else:
            print(key, value)
    for key, value in result["p2"].items():
        print(key, [(p["pair"], round(p["difference"], 4), np.round(p["ci"], 4).tolist()) for p in value["top_region_pairs"][:3]])


if __name__ == "__main__":
    main()
