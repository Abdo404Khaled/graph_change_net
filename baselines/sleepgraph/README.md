# sleep-graph-benchmark (shared upstream of the graph baselines)

Juvina & Nicolae, "Improving graph-based sleep staging with shared waveform node
representations", *Machine Learning and Knowledge Extraction* 8(9):289, 2026.
Code: <https://github.com/simona-juvina/sleep-graph-benchmark>, BSD-3, cloned at commit
`51440c3` into `upstream/` by `scripts/baselines_setup.sh` (not committed). It re-implements
GraphSleepNet, ProductGraphSleepNet, MSTGCN, JK-STGCN and STDP-GCN with their native frontends
and already prepares ANPHY. The model folders `graphsleepnet`, `graphsleepnet_distance`,
`mstgcn` and `productgraphsleepnet` use it; nothing in `upstream/` is changed.

The MDPI paper itself could not be downloaded automatically; put its PDF here as `paper.pdf`
if needed.

## Native configuration on ANPHY (upstream defaults, read from the code)

| | Upstream default | Where |
|---|---|---|
| Channels | 19 EEG (Fp1 Fp2 F3 F4 F7 F8 C3 C4 Cz P3 P4 O1 O2 T3 T4 T5 T6 Fz Pz) **plus EOG1, EOG2, ChEMG1, ChEMG2**: `--use-eog` and `--use-emg` default to true although the docstring says 19 | `preprocessing/anphy.py:312-335, 1124-1145` |
| Filters | Butterworth order 4, zero-phase, at the original rate: EEG 0.3-100 Hz, EOG 0.1-100 Hz, EMG 10-100 Hz; no notch, no re-reference | `preprocessing/filters.py:10-49`, `anphy.py:964-1007` |
| Rate | MNE FFT resampling to 100 Hz | `anphy.py:1009-1013` |
| Epochs | 30 s on the hypnogram grid; unscored and lights-off epochs dropped; continuity segments recorded | `anphy.py:1021-1065` |
| Context | centred windows: 5 epochs (GraphSleepNet, MSTGCN), 9 (ProductGraphSleepNet); `pad_mode=drop` leaves the first and last 2 (4) epochs of every continuity segment unpredicted | `data/recordings.py:290-399` |
| Node features | GraphSleepNet, ProductGraphSleepNet: differential entropy in 9 bands (0.5-4, 2-6, 4-8, 6-11, 8-14, 11-22, 14-31, 22-40, 31-50 Hz) from one Hann-windowed FFT of the 30 s epoch, z-scored per channel and band on the training nights. MSTGCN: its released two-branch CNN, 256 features per channel, trained first | `models/encoders.py:10-56`, `models/mstgcn.py:18-57` |
| Selection | validation macro-F1 smoothed over 5 epochs, eligible from epoch 24, patience 10; ReduceLROnPlateau | `training/selection.py` |
| Training | Adam 1e-3, batch 64, up to 200 epochs (GraphSleepNet, ProductGraphSleepNet); MSTGCN: feature net Adam 1e-4, then graph net Adam 1e-5 with a domain-adversarial head; seed 42 + fold | `training/*.py` |

## What our runners change, and why

- **Fold:** our fold 0 (`baselines/folds/fold0.json`) is written in upstream's split format
  and passed with `--split-file`.
- **Test predictions:** our scorer needs every scored test epoch. Training keeps upstream's
  `pad_mode=drop`; the test predictions are exported with upstream's `--pad-mode edge`, which
  repeats the edge epoch of a segment so that every epoch gets a window.
- **Labels:** upstream parses the hypnogram text files itself. The runner checks that its
  epochs line up with our scored epochs (same onset and stage) before scoring.
