# Baselines

Published sleep-staging models, run on ANPHY-Sleep twice: once under each paper's own
configuration (**native**), and once under **our** configuration. Every run uses the same
fold and is scored by the same code as our model.

## Layout

```
baselines/
  settings.yaml          reference run, fold file, output and cache folders, smoke-test sizes
  folds/fold0.json       train 18 / val 5 / test 5 nights, copied from runs/exp15_final_fold0
  common/
    reference.py         reference config, fold, the exact scored epochs
    native.py            EDF channels / bipolar pairs (our name matching), our epoch onsets and labels
    ours.py              our data protocol (data.build_epoch_loaders) + our stage-2 trainer (train.classify)
    scoring.py           one prediction per scored test epoch; evaluate.scores; clean subset; per night
  run.py                 python -m baselines.run <model> <native|ours> [--quick]
  summary.py             python -m baselines.summary   (all results as one table)
  <model>/
    README.md            paper summary, native configuration with file:line sources, decisions
    paper.pdf            the paper (local only)
    upstream/            the authors' code at a pinned commit (local only; commit in README)
    model.py, runner.py  our adapter around their code
    native.yaml, ours.yaml
```

Models: `anysleep`, `usleep`, `deepsleepnet`, `cbramod` and `hypnopc` are runnable. `mtssrl_md`
documents a paper with ANPHY results but no released code.

Graph baselines: `graphsleepnet`, `graphsleepnet_distance`, `mstgcn` and `productgraphsleepnet`
are the published graph sleep models (one electrode graph per 30 s epoch). They share one
upstream, sleep-graph-benchmark, cloned into `sleepgraph/upstream/`; `sleepgraph/` holds the
shared code (`native.py`: upstream's preparation and trainer on our fold, `native_predict.py`:
every scored test epoch from the selected checkpoint, `ours.py`: their networks on our runs)
and its README lists the native ANPHY settings. Each graph run also writes `graphs.npz`, the
electrode graph of every scored test epoch. The ablations of our own graph hierarchy are in
`../ablations/`.

This package never imports `components.py` or `model.py`. It reuses, read-only:

- `data.py`: hypnogram, channel matching, cached epochs, loaders;
- `train.classify`;
- `evaluate.scores` and `evaluate.predict_proba`.

`config.py` imports `components.check_model` at module level, so loading a config imports that
module, but nothing in it is used.

## The two configurations

| | native | ours |
|---|---|---|
| Channels and derivations | as in the authors' code (often EEG + EOG; see each README) | the 83 montage EEG electrodes, nothing else |
| Sampling rate | the authors' (100, 128 or 200 Hz, resampled from 1 kHz with `resample_poly`) | 1000 Hz, never resampled |
| Filtering, reference, scaling | the authors' | notch 60 Hz, 0.5–45 Hz, common average; per-electrode robust scaler fitted on the training nights, clipped to ±20 (our cache) |
| Window | the authors' (whole night, or 25, 35 or 20 epochs) | runs of 20 epochs; each test epoch is predicted from the run that ends at it |
| Training | the authors' recipe, or their released weights zero-shot | our stage 2 unchanged (`train.classify` with the reference run's settings). Networks trained from scratch get Exp 15a's setting (no frozen probe); pretrained ones get Exp 15's 4-epoch probe |
| Architecture | the authors' | the authors', changed only where the input forces it (input channels, sampling-rate-dependent sizes); see each README |

**Common to both:**

- **Fold:** fold 0 of Exp 15. All five folds share the same 5 test nights, and Exp 15
  moved ±0.008 κ across folds.
- **Scoring:** every one of the **4,462 scored test epochs**. The scorer refuses a
  prediction set that is not exactly this set.
- **Reference:** Exp 15 fold 0, κ 0.8311, macro-F1 0.8430. It is read from its run
  directory, not rerun.

## Running

```bash
python -m baselines.run anysleep native         # one baseline, one configuration
python -m baselines.run anysleep ours --quick   # smoke test: few nights, items and steps
python -m baselines.summary                     # table of every result, reference first
```

Outputs go to `baselines/results/<model>/<config>/`:

- `config.yaml`
- `predictions.npz`
- `metrics.json`: test, test_clean, per night, parameters.

Native preprocessing is cached in `baselines/cache/<model>/<config>/`. Smoke tests use
`*_quick` folders, which are deleted after testing.

## Caveats that apply to several baselines

- **Receptive field at 1 kHz.** U-Sleep and AnySleep keep their kernels sized in samples, so
  under our configuration their receptive field in seconds shrinks about 8× (U-Sleep 544 s →
  about 70 s; AnySleep about 14 min → about 106 s). The released architecture is the default.
  Depth and kernel are YAML values, so a receptive-field sensitivity run is a configuration
  change.
- **Training/test mismatch for bidirectional models.** DeepSleepNet's BiLSTM and CBraMod's
  sequence encoder are bidirectional. Within our runs the scored test epoch is the last
  position, so no future input reaches it. Training, however, labels every position.
- **Zero-shot native runs.** AnySleep and U-Sleep (native) use released weights trained on
  other datasets. Their native setting is zero-shot on ANPHY, as in the AnySleep paper's
  Table S5.
