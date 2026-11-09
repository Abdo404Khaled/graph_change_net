# AnySleep baseline

This folder documents the AnySleep baseline and holds a read-only copy of the upstream repository.
No upstream code has been installed or executed. The only things run against `upstream/` were
`git clone`, `git rev-parse` and a `torch.load(..., weights_only=True)` on the checkpoints to read
their tensor shapes and parameter counts.

Upstream file references use the form `path:line` relative to `upstream/`. The paper is cited by
section number in `paper.pdf` (arXiv v3). Anything not read directly from code or paper is marked
**inferred**.

## 1. Citation and links

| Item | Value |
|---|---|
| Paper | Grieger N., Raskob J., Mehrkanoon S., Bialonski S. *AnySleep: a channel-agnostic deep learning system for high-resolution sleep staging in multi-center cohorts.* npj Digital Medicine 9, 829 (2026) |
| DOI | 10.1038/s41746-026-03406-7 |
| arXiv | 2512.14461v3 (31 Oct 2026), saved as `paper.pdf` (30 pages, 3.06 MB) |
| Repository | https://github.com/dslaborg/AnySleep |
| Commit cloned | `899bfff284f3eb3e7448bc90990f2230757470a2` (2026-11-02, "drop git LFS files from tracking") |
| License | MIT, Copyright (c) 2026 dslaborg (`LICENSE`) |
| Results archive | Zenodo 10.5281/zenodo.23764134. It holds `logs/` and `examples/output/`, including the ANPHY prediction files. Not downloaded. |
| Data preparation | Done by a separate repository, https://github.com/dslaborg/Sleep-Datasets, which writes the HDF5 files used for training. Not cloned. |

## 2. Paper summary

**Problem.** Sleep staging models usually expect a fixed montage, and the 30 s scoring epoch is a
convention rather than a physiological unit. AnySleep accepts any number and any combination of
EEG/EOG channels and can output stages at sub-30 s resolution.

**Method.** The backbone is U-Sleep: a 1-D U-Net with 12 encoder blocks, a connector and 12 decoder
blocks. AnySleep runs every channel through the shared encoder on its own. At each of the 13 skip
connections (12 encoder levels plus the connector), a channel-attention module fuses the channels:
it averages each channel's feature map over time, passes it through an MLP (hidden size 40, then
BatchNorm, ReLU and one output unit), applies a softmax across channels, and takes the weighted sum.
The decoder therefore works on a single fused stream. A segment classifier average-pools the decoder
output into windows; the pooling width sets the output resolution, with 3840 samples = 30 s used in
training. The paper also tests early-fusion and late-fusion variants, which use one attention module
plus a channel encoder (§4.2). The model has 3,157,856 parameters, against 3,114,337 for U-Sleep.

**Data.** More than 20,000 recordings from 28 datasets. The split is 18,038 training, 559 validation
and 1,992 test recordings, and test is divided into in-distribution, 30 s hold-out and
high-frequency hold-out groups (§4.1). ANPHY was used **only as a held-out test set**: there was no
training or fine-tuning on ANPHY, so these are zero-shot results.

**Main results.** Weighted mean MF1 on the in-distribution test sets: AnySleep 0.812, late fusion
0.804, U-Sleep 0.799, early fusion 0.787. On the hold-out sets: 0.777, 0.774, 0.768 and 0.752
(§2, Table 1). The high-frequency outputs pick up arousal-like wake intrusions and help predict
OSA, NT1 and insomnia.

**ANPHY (Table S5).** 28 recordings, mean ± std over 3 training runs:

| Model | W | N1 | N2 | N3 | R | MF1 | Cohen's κ |
|---|---|---|---|---|---|---|---|
| AnySleep | 0.91 (.002) | 0.46 (.009) | 0.90 (.000) | 0.91 (.005) | 0.88 (.002) | 0.81 (.002) | 0.82 (.002) |
| U-Sleep | 0.89 (.010) | 0.47 (.001) | 0.89 (.005) | 0.87 (.019) | 0.87 (.002) | 0.80 (.004) | 0.79 (.006) |

The caption says "83 EEG and 2 EOG channels … 166 evaluations of different EEG-EOG channel pairs".
The code and log show **79 EEG + 2 EOG = 81 channels and 158 pairs** (see §4 below).

## 3. Native configuration

| Setting | Value | Source |
|---|---|---|
| Input channels | Any number of EEG and/or EOG channels; EMG/ECG are not used | paper §2, §4.2; `base/data/base_dataset.py:128-140` |
| Channels at train time | For each batch, draw n ∈ {1..N_ch} with p_n = (Σ_i n/i)^-1, so p_n ∝ 1/n. With N_ch = 10, p = .341, .171, .114, … .034 and E[n] = 3.41. Each sample then takes n of its recording's channels uniformly, regardless of type, with replacement if the recording has fewer than n | `base/data/anysleep_train_dataset.py:112-116, 171-177`; paper §4.3 eq. 2. The docstring at `:41-42` says it "favor[s] having more channels"; the code favors fewer |
| Channels at inference | All available EEG+EOG in **one** forward pass; channel order is shuffled, and the model does not depend on order | `base/data/anysleep_eval_dataset.py:196-204`; paper §4.4 |
| Sampling rate | 128 Hz | `config/base_config.yaml:22`; `examples/anysleep_no_hydra.py:73` (hard-coded) |
| Resampling | `scipy.signal.resample_poly(data, 128, fs)`, polyphase | `examples/anphy/predict_anphy.py:86-87`; paper §4.1 |
| Filtering | None. No bandpass or notch ("did not yield notable performance improvements") | paper §4.1; no filter in `predict_anphy.py:65-96` |
| Re-referencing | None. Channels are used as stored in the file | `predict_anphy.py:26-36` |
| Clipping | Example code: per channel, clip the **raw** (resampled) signal to ±20·IQR around 0, *before* scaling. Paper: clip the *normalized* signal to [-20, 20]. These are equivalent when the channel median is ≈0 (inferred) | `predict_anphy.py:39-62, 91`; paper §4.1 |
| Normalization | `sklearn.preprocessing.RobustScaler` (median/IQR) fitted **per recording, per channel** on the whole night | `predict_anphy.py:93-94`; paper §4.1 |
| Units | Volts as returned by MNE; the µV conversion is commented out. RobustScaler makes units irrelevant | `predict_anphy.py:34` |
| Epoch length | 30 s = 3840 samples | `config/base_config.yaml:23`; `anysleep_no_hydra.py:74` |
| Training sequence | 35 consecutive epochs = 134,400 samples (17.5 min) | `config/exp002/anysleep_config.yaml:20`; `anysleep_train_dataset.py:132`; paper §4.3 |
| Inference sequence | The whole night in one forward pass, with no windowing | `predict_anphy.py:306-332`; paper §4.4 |
| Receptive field | 14.36 min (110,318 samples including the 3840 pool; backbone alone 106,479 samples = 13.9 min). Computed by us; it matches the paper | paper §4.3 |
| 30 s prediction | Segment classifier: Conv1d(6→6, k=1), Tanh, AvgPool1d(3840, stride 3840), Conv1d(6→5, k=1), ELU, Conv1d(5→5, k=1), giving logits; argmax per epoch. The pool has no parameters, so `sleep_stage_frequency` (pool = 3840/f) can change without retraining | `base/model/usleep_base_components.py:129-145`; `anysleep_no_hydra.py:96, 262-273` |
| U-Net depth | 12 encoder + connector + 12 decoder blocks | `config/exp002/anysleep_config.yaml:53`; `base/model/anysleep.py:85-101` |
| Filters | `calc_filters(1, 12, complexity=1.2923, scale=1.4142)`: encoder 1→6,9,11,15,20,28,40,55,77,108,152,214; connector 302; decoder 214,152,108,77,55,40,28,20,15,11,9,6 | `base/model/utilities.py:4-42`; `anysleep.py:73-75` |
| Encoder block | Conv1d(k=9, same), ELU, BatchNorm, MaxPool(2,2); odd lengths are left-padded by 1 | `usleep_base_components.py:25-51` |
| Decoder block | Nearest Upsample×2, Conv1d(k=2, same), ELU, BN, center-crop to skip, concat, Conv1d(k=9, same), ELU, BN | `usleep_base_components.py:70-107` |
| Attention (13 modules) | Mean over time, Linear(F→40), BN1d, ReLU, Linear(40→1), softmax over channels, weighted sum. **One scalar weight per channel per sample, for the whole input length** | `base/model/anysleep.py:168-259`; paper §4.2 |
| Init | Kaiming normal (fan_out), zero bias, BN weight 1 | `base/model/base_model.py:16-53` |
| Parameters | 3,157,856 (AnySleep), 3,114,337 (U-Sleep), 3,131,601 (early fusion), 3,137,356 (late fusion). Reproduced from the checkpoints | paper §4.2 |
| Optimizer | Adam with AMSGrad, betas (0.9, 0.999), weight decay 0 | `config/exp002/anysleep_config.yaml:66-70` |
| LR | 1e-5 fixed. The CyclicCosineDecayLR scheduler has min multiplier 1.0, so the rate stays constant | `anysleep_config.yaml:57-65, 79`; paper §4.3 |
| Batch size | 64, built inside the dataset so that every sample in a batch has the same channel count. Late fusion uses 32 | `anysleep_config.yaml:13, 23`; `config/exp004/exp004a.yaml:9-12` |
| Epoch definition | 28,571 samples/epoch, rounded up to 447 batches (28,608 samples). The paper says 443 updates and ≈10^6 sleep epochs | `anysleep_config.yaml:21`; `anysleep_train_dataset.py:92-98` |
| Max epochs / early stopping | 10,000 / 100 epochs without improvement in validation MF1. The score is MF1 over all validation epochs pooled ("concat"), with no majority voting | `anysleep_config.yaml:73-75, 88-105`; `base/training/clas_trainer.py:129-146`; `base/results/sleep_stage_tracker.py:138-149` |
| Gradient clipping | Off (0) | `anysleep_config.yaml:74` |
| Sampling scheme | Draw a uniform stage. Draw a dataset with p = 0.5·(1/N_d) + 0.5·(N_rec,d/ΣN_rec). Draw a uniform recording, then a uniform epoch of that stage, and place it at a uniformly random position in the 35-epoch window. Retry up to 100 times if the stage is absent | `anysleep_train_dataset.py:144-190, 192-229, 234-305`; paper §4.3 eq. 1 |
| Loss | Unweighted `nn.CrossEntropyLoss` averaged over every epoch in the window; label 9 (artifact/unknown) is masked | `clas_trainer.py:17, 200-205`; paper §4.3 |
| Class weighting | None. Class balance comes only from the stage-stratified sampling | `clas_trainer.py:202` |
| Augmentation | None ("masking-based augmentations … did not improve") | paper §4.3 |
| Batch of size 1 | **Skipped** during training, because of BatchNorm | `clas_trainer.py:190-192` |
| Seeds | `seeds: [null, null, null]`, so each run is random. The 3 runs were started by the same command | `anysleep_config.yaml:6`; `config/exp002/run.md:4-6` |
| Training split | Lists of U-Sleep-style subjects (13 datasets for train/val; ANPHY is absent) | `config/usleep_split.yaml` |
| Evaluation metric | Per-stage F1 from TP/FP/FN pooled over all epochs of a dataset or recording; MF1 is their unweighted mean. A stage that is absent and never predicted gets F1 = 1 | `sleep_stage_tracker.py:245-283`; paper §4.4 eq. 3 |
| Evaluation prediction | `softmax(logits).argmax` | `base/evaluation/clas_evaluator.py:83` |
| U-Sleep inference | Every (EEG, EOG) pair is run separately; per-epoch argmax, then majority vote with random tie-breaking | `examples/anphy/predict_anphy_usleep.py:99-133, 347-396` |

## 4. ANPHY specifics, from `examples/anphy/*`

- **EDF reading.** `mne.io.read_raw_edf(path, include=input_channels)`, then `raw.get_data()`
  (`predict_anphy.py:26-32`). Files come from `glob("/mnt/data/anphy/*/*.edf")` (`:105`), which
  includes `EPCTL01 - fixed.edf`.
- **Channel list.** `predict_anphy.py:111-281`. The list contains both bare names (`Fp1`) and `-Ref`
  names (`Fp1-Ref`) because EPCTL21 and EPCTL25, and partly EPCTL07, label channels `X-Ref`. It
  also lists the misspellings `F4-` (EPCTL05) and `C1-` (EPCTL06). `include` keeps whichever names
  exist.
- **SO1, SO2, ZY1 and ZY2 are excluded.** They are commented out at `predict_anphy.py:132-133,
  136-137, 218-219, 222-223` and at the same places in `predict_anphy_usleep.py:169-174, 253-258`.
  The log confirms **81 channels per recording (79 EEG + EOG1 + EOG2)** for all 28 nights
  (`predict_anphy_log.txt`). The U-Sleep log prints **158** pairs (79 × 2), not 166. Paper Table S3
  lists SO1/SO2/ZY1/ZY2 as EEG, and the Table S5 caption says 83 EEG and 166 pairs. That caption
  does not match the code that produced the logs.
- **Not used.** EMG (ChEMG1/2), leg leads (RLEG±/LLEG±) and ECG1/2 are not in the include list.
- **Re-referencing.** None. The data stay in the recording reference.
- **Preprocessing.** `resample_poly` 1000→128 Hz, then the ±20·IQR clip, then RobustScaler per
  channel over the whole night (`predict_anphy.py:65-96, 297`).
- **Inference.** The whole night goes in as one tensor `(1, T, 81)` with one forward pass per
  checkpoint (`anysleep-run1..3`) and `sleep_stage_frequency=1`. The output is
  `argmax(logits)` per 30 s epoch (`:283-332`). AvgPool floors the length, so a trailing partial
  epoch is dropped. Results are saved to `output/anphy/predictions_c.npz` (`:336`), but
  `table_s5.ipynb` reads `predictions_anysleep.npz`. The file name differs and the content is
  presumably the same (inferred).
- **U-Sleep on ANPHY.** Every (EEG, EOG) pair, batches of 8 pairs, whole night, majority vote
  (`predict_anphy_usleep.py:347-396`).
- **Hypnogram parsing.** `scripts/final-figures/table_s5.ipynb`, cell 1. Each line of the txt file
  is `stage<TAB>onset_s<TAB>30`, and every duration must equal 30 (assert). If the first onset is
  not 0, `onset/30` epochs of `UNK` are prepended. Gaps of more than 30 s are filled with the
  **previous** stage. The map is `W:0, N1:1, N2:2, N3:3, R:4, L:9, UNK:9`. The ground-truth path is
  a placeholder (`FIXME`). Locally, EPCTL01's annotation file is named `test1.txt`, so the authors
  must have renamed or arranged the files themselves (inferred).
- **'L' (lights-on/off) epochs.** Mapped to 9. They are still **fed to the model**, because the
  whole recording is the input, but they are **removed from the metrics** (cell 3, `art_mask`).
  There are 1,807 `L` epochs across the 28 local annotation files.
- **Length alignment.** If the ground truth is longer than the prediction, it is truncated to the
  prediction length (cell 2).
- **Metrics.** Every scored epoch of all 28 nights is pooled per run:
  `f1_score(labels=range(5), average=None)`, MF1 is the mean of the five, and `cohen_kappa_score`
  is unweighted. The table reports the mean and std over the 3 runs (cells 3-4). These are pooled
  epoch-level metrics, not per-subject averages.

## 5. Model interface

- **Standalone, no Hydra:** `examples/anysleep_no_hydra.py` (identical to
  `examples/anphy/anysleep_no_hydra.py`). Its only dependencies are numpy and torch.
  ```python
  from anysleep_no_hydra import AnySleep
  model = AnySleep(path=None, depth=12, complexity=1.2923, scale=1.4142,
                   hidden_size=40, sleep_stage_frequency=1)
  ```
  The sampling rate is **hard-coded to 128** (`:73`), so the pool is 3840/f. The constructor
  calls `load(path)`, i.e. `torch.load(weights_only=True)["state_dict"]` (`:151-156`).
- **Hydra version:** `base.model.anysleep.AnySleep(path, seed, depth, complexity, scale,
  hidden_size, sleep_stage_frequency, save_att_weights)`. It needs `base.config.Config` to be
  initialized, because the pool size comes from `cfg.data.sampling_rate * cfg.data.epoch_duration`
  (`usleep_base_components.py:132-136`), and `load` resolves the path against
  `cfg.general.snapshot_dir`. The early- and late-fusion classes are `AnySleepEF` and `AnySleepLF`,
  but `config/exp003|exp004/anysleep_config.yaml` target `...anysleep_early_fusion.AnySleep`, a name
  those modules do not define (inferred: these configs would not instantiate as written).
- **Input:** float32 `(batch, time_samples, channels)`. The channel count is the last dimension. It
  can be anything ≥1 and may change between batches, but it is fixed within a batch. There is no
  channel identity or position input, and the model is permutation-invariant. `time` should be a
  multiple of the pool size (3840 at f=1); otherwise the remainder is dropped.
- **Internals:** `(B, T, C)` is reshaped to `(B·C, 1, T)` for the encoder. The skip maps are
  `(B, C, F, T_i)`, attention reduces them to `(B, F, T_i)`, and the decoder runs on the fused
  maps.
- **Output:** logits `(batch, n_epochs, 5)` in the order W, N1, N2, N3, REM.
- **Fine-tuning:** there is no dedicated script. `scripts/train.py -cn=exp002/exp002a
  model.path=<ckpt>` loads the weights in `__init__` (`base/model/base_model.py:140-156`) and then
  trains, which amounts to fine-tuning (inferred). It needs the Sleep-Datasets HDF5 layout and a
  split YAML. The standalone file has no training code.
- **Requirements** (`requirements.txt`): `hydra-core==1.4.0.dev1`, `hydra-joblib-launcher==1.2.0`,
  `hydra-nevergrad-sweeper==1.3.0.dev0`, plus unpinned `setuptools`, `numpy`, `scipy`,
  `scikit_learn`, `h5py`, `mne`, `lspopt`, `openpyxl`, `matplotlib`, `tabulate`, `seaborn`,
  `jupyter` and `torch` (install manually). There is no pyproject, setup.py or Python version pin.
  Inference with the standalone file plus the example preprocessing needs only
  torch/numpy/scipy/scikit-learn/mne, which are already in this repo's environment.

## 6. Running it in our configuration (assessment only; nothing implemented)

Our protocol: 83 EEG channels at 1000 Hz with no resampling; 0.5–45 Hz bandpass, 60 Hz notch and
common average reference (CAR); a per-channel RobustScaler fitted on training subjects, clipped to
±20; runs of 20 consecutive 30 s epochs (600,000 samples per channel); causal, so epoch t is
predicted from epochs t-19..t.

**What has to change in the code**

1. **Pool size.** Set `data.sampling_rate: 1000` (`config/base_config.yaml:22`), or change
   `anysleep_no_hydra.py:73` to 1000. The pool becomes AvgPool1d(30000), and 600,000 / 30,000 = 20
   outputs exactly. The pool has no weights, so **all AnySleep/U-Sleep checkpoints still load**;
   their state dicts contain no pool parameters (checked). For the fusion variants, the
   channel-encoder kernel is `sampling_rate//2` with stride `//4`
   (`base/model/anysleep_early_fusion.py:205`). At 1 kHz that becomes 500/250, which changes the
   shapes, so the EF/LF checkpoints would not load.
2. **Training window.** `window_size: 20` and the 1000 Hz sampling rate give a window of
   `20·30·1000` samples (`anysleep_train_dataset.py:132, 286-287`). The upstream sampler places the
   target epoch at a random position in the window. For a causal model it would have to be the
   last position, and the loss restricted to the last epoch (or kept over all 20).
3. **Preprocessing.** Their per-recording median/IQR scaler fitted on the whole night is replaced
   by ours: train-subject RobustScaler, bandpass, notch and CAR. Upstream applies none of the
   filters or the CAR. Their HDF5 pipeline (Sleep-Datasets) is not used; we would feed tensors
   directly.
4. **Batch ≥ 2.** `clas_trainer.py:190-192` silently skips batches of size 1.
5. **Divisibility.** No hard constraint. Odd lengths are left-padded and the decoder center-crops.
   600,000 = 2^6·9375, so padding happens at levels 6 and 11 and the bottleneck length is 147.
   Natively, 134,400 pads at levels 8 and 9 with a bottleneck of 33. The pool needs
   `30000 / sleep_stage_frequency` to be an integer.

**Receptive field at 1 kHz.** The backbone's receptive field is fixed at 106,479 **samples**. At
128 Hz that is 832 s; at 1000 Hz it is only **106.5 s**, or 136.5 s including the 30 s pool.
Convolutions use `same` padding and are centered, so each sample sees ±53 s. The prediction for the
last epoch of a 20-epoch window therefore uses only about the previous 1.8 epochs, not 19. Matching
the native ~14 min needs depth 15: receptive field 852 s, **24.55 M parameters** (7.8×), bottleneck
of 851 filters at length 19. Depth 13 gives 213 s and 6.23 M; depth 14 gives 426 s and 12.35 M.
Covering the full 600 s causal history from the last epoch (half-RF ≥ ~585 s) would need depth 16,
with 48.9 M parameters. Our parameter formula reproduces 3,157,856 at depth 12. A larger first
pooling stride, for example 8, would restore the receptive field at depth 12. But it amounts to a
learned decimation, which is a judgment call against our no-resampling rule.

**Pretrained weights at 1 kHz.** The kernels were learned at 128 Hz. At 1 kHz a 9-tap kernel spans
9 ms instead of 70 ms, which shifts every learned frequency band up by 7.8×. Zero-shot use at 1 kHz
is therefore not meaningful (inferred). The options are training from scratch at 1 kHz, or
fine-tuning, which re-learns most of the encoder. A native-faithful reference point is still
cheap: resample to 128 Hz and run the released checkpoints zero-shot, as in §4. That violates our
no-resampling premise, so it can only serve as a reference, not as the comparison.

**Causality caveats.** (a) Each attention weight is one scalar per channel per input, computed from
the time-mean over the whole input. Within a 20-epoch causal window it uses only the past, but
natively it averages the whole night. (b) The `same` padding zero-pads the right edge, so the last
epoch, the only one scored, sits on an edge effect. (c) Scoring every epoch causally needs one
600,000-sample forward pass per epoch, a stride-1 sliding window, instead of the native single pass
per night.

**Memory and compute** (analytic, inferred; float32, no AMP, no checkpointing). For batch 1,
83 channels × 600,000 samples: about 19.1·L feature values per channel across the 12 encoder
levels. With ~3.5 saved tensors per level (BN in/out, pool output, max-pool indices) that comes to
about **13.4 GB of encoder activations for training**. The fused decoder adds about 0.3 GB and the
input is 0.2 GB, so expect **~14–16 GB per window** including allocator overhead. On an 80 GB A100
that allows about 4 windows per batch. Batch 1 is not usable as-is (point 4 above). The forward
pass costs about 107 GMAC per window. That is close to one native training batch (64 × 134,400
samples × E[n] = 3.41 channels, ~119 GMAC, ~12.5 GB activations). Inference without gradients
peaks at about **3–4 GB** per window, dominated by level-0 tensors `83 × 6 × 600,000` at 1.2 GB
each. For scale, the native whole-night ANPHY pass (81 channels, ~8 h at 128 Hz) peaks at an
estimated ~19 GB without gradients. With depth 15 the deeper levels are short, so activation memory
grows only slightly, but the parameter count grows 7.8×. Using all 83 channels instead of
upstream's 79+2 adds SO1/SO2/ZY1/ZY2, whose behaviour natively looks closer to EOG (inferred).
Since the model ignores channel identity, that only changes the channel count.

**Risks.** The protocol differs in five ways at once (sampling rate, filtering/CAR, scaler fit,
window/causality, channel set), so a gap against Table S5 cannot be attributed to any one of them.
Our scores are per-fold on 5 test subjects, while Table S5 is zero-shot on all 28 nights with
pooled metrics, so the numbers are not directly comparable. The published ANPHY result is zero-shot
from a model trained on more than 18k nights. A model trained only on ANPHY training subjects has
roughly 20 nights, which is a very different data regime.

## 7. Files

| Path | Content |
|---|---|
| `paper.pdf` | arXiv 2512.14461v3, downloaded from https://arxiv.org/pdf/2512.14461 |
| `upstream/` | `git clone` at `899bfff`, a nested git repo of 311 MB (140 MB of it is `.git`). Not a submodule |
| `upstream/examples/anysleep_no_hydra.py`, `usleep_no_hydra.py` | Standalone models |
| `upstream/examples/anphy/` | `predict_anphy.py`, `predict_anphy_usleep.py`, their logs, and copies of the standalone models |
| `upstream/scripts/final-figures/table_s5.ipynb` | ANPHY ground-truth parsing and metrics |
| `upstream/base/` | Hydra framework: models, datasets, trainer, evaluator, trackers |
| `upstream/config/exp00{1..4}/` | U-Sleep, AnySleep, early-fusion and late-fusion configs, with `run.md` holding the exact commands |

Checkpoints in `upstream/models/` are real files, not LFS pointers. Each is a torch zip with
`{"state_dict": ...}`:

| File | Size (bytes) | Parameters |
|---|---|---|
| `anysleep-run1.pth`, `-run2`, `-run3` | 12,763,675 each | 3,157,856 |
| `usleep-run1.pth`, `-run2`, `-run3` | 12,551,209 each | 3,114,337 |
| `anysleep-early-fusion-run1.pth`, `-run2`, `-run3` | 12,643,975 each | 3,131,601 |
| `anysleep-late-fusion-run1.pth`, `-run2`, `-run3` | 12,656,923 each | 3,137,356 |

## 8. How to run and implementation decisions

```bash
.venv/bin/python -m baselines.run anysleep native [--quick]
.venv/bin/python -m baselines.run anysleep ours [--quick]
```

Outputs go to `baselines/results/anysleep/<config>[_quick]/` (`config.yaml`, predictions,
`metrics.json`). Caches go to `baselines/cache/anysleep/<config>[_quick]/`.

Files: `native.yaml` and `ours.yaml` (all settings), `model.py` (networks), `runner.py`
(`run(spec, settings, out_dir, quick)`). Upstream code is imported from
`upstream/examples/anphy` through `sys.path` and is never modified.

**Status: not yet executed.** Running either configuration imports and executes upstream code
(`anysleep_no_hydra.py`, `predict_anphy.py`). That was refused by this session's permission
policy, so the smoke tests, shapes, memory and timings in §6 are not measured yet. The two
modules were read in full before use. At import time they do nothing but import numpy, torch,
scipy, scikit-learn and mne; `predict_anphy.main()` is behind `if __name__ == "__main__"`.

### native (decisions of the coordinator)
- **Zero-shot.** Uses `anysleep-run1/2/3.pth`, exactly as `examples/anphy/predict_anphy.py`.
  There is no training, so the training nights are unused.
- **Channels:** the 79 EEG channels of that script, with SO1, SO2, ZY1 and ZY2 left out as the
  code does (the paper says 83), plus EOG2 and EOG1. They are listed in `native.yaml` in the
  EDF order of upstream's log. Channels are read with our name matching, so `-Ref` suffixes and
  `F4-`/`C1-` resolve, in µV, one at a time.
- **Preprocessing:** `resample_poly(x, 128, sfreq)` channel by channel, which equals upstream's
  `axis=1` call. Then upstream's own `clip_noisy_values` (20·IQR), then `RobustScaler` fitted on
  that night per channel. No filter and no re-referencing. Clipping and scaling are both
  relative to each channel's IQR, so working in µV instead of V makes no difference.
- **Inference:** the whole night is one `(1, samples, 81)` tensor, one forward pass per
  checkpoint, using upstream `AnySleep` from `anysleep_no_hydra.py`.
- **Alignment:** output `k` pools seconds `[30k, 30k+30)`. Every onset in our hypnograms is a
  multiple of 30 s from the recording start (checked on all 28 nights), so a scored epoch at
  `onset` takes output `onset/30`. The runner raises an error if an onset is off-grid or has
  no output.
- **Scoring and summary:** one predictions file per checkpoint, each scored by `score_run` on
  the 5 test nights (4,462 scored epochs). `metrics.json` holds each run, mean ± population
  sd over the 3 runs (as `table_s5.ipynb` reports), and the softmax-averaged ensemble of the 3
  checkpoints as an extra row. Our scorer pools the test epochs of our split, so the scores
  are not those of Table S5 (all 28 nights).
- **Quick mode:** only the first `quick.test_subjects` test night, whole and scored.

### ours (decisions of the coordinator)
- **Architecture:** as released. Depth 12, kernel 9, released filter counts (`calc_filters`
  with complexity 1.2923 and scale 1.4142), 13 channel-attention modules with hidden size 40.
  The only change is the segment classifier's average pool, which is **30,000 samples**
  (30 s at 1 kHz) instead of 3,840. Depth, kernel and pool are YAML values, so a
  receptive-field sensitivity run (§6) is only a config change.
- **Modules:** `AnySleepBackbone` subclasses upstream `AnySleep`, builds it from the upstream
  blocks with the configurable kernel and pool, and inherits `forward` unchanged. With depth
  12, kernel 9 and pool 3840 the released checkpoints would load into it (not yet run).
- **Parameters:** 3,157,856. The pool has no weights, so the count equals the released model's
  (computed analytically, §6).
- **Initialisation:** upstream's training initialisation (`_weights_init`, Kaiming fan-out),
  copied into `model.py` because `base/model/base_model.py` imports Hydra. Built after
  `torch.manual_seed(42)`, the reference run's seed for fold 0.
- **Input:** our `[B, 20, 83, 30000]` is laid end to end as upstream's `(B, 20·30000, 83)`.
  The output is `(B, 20, 5)`, one row per epoch. The attention weights and the `same`
  convolutions see the whole 20-epoch run. Test items end at the scored epoch, so each test
  prediction uses only that epoch and the 19 before it.
- **Training from scratch:** our stage-2 trainer (`train.classify`) and loaders through
  `run_ours`, with the reference run's settings. The 128 Hz weights do not transfer to 1 kHz
  (§6).
  - The only override is `finetune.freeze_epochs: 0` (no frozen-head phase, as Exp 15a).
  - The trainer's head is upstream's `segment_classifier`: Conv1d → Tanh → AvgPool → Conv1d →
    ELU → Conv1d, 107 parameters. It is trained at `finetune.lr`; everything else at
    `lr · backbone_lr_scale` (0.1).
  - Our class-balanced focal loss, label smoothing, EMA weights, augmentations and batch size
    (`finetune.batch_size: 2`) apply unchanged. Upstream's trainer, AMSGrad at 1e-5 and its
    stage-stratified sampler are not used; that is the point of "ours".
- **Memory estimate (analytic, §6):** about 14–16 GB per item of 20 × 83 × 30,000 samples, so
  about 28–32 GB for a batch of 2 in fp32.
