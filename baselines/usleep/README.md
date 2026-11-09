# U-Sleep baseline

This folder documents the U-Sleep baseline, holds a read-only clone of the official repository and
runs the baseline (§8). The official U-Time code was never installed or executed. The runner
imports one AnySleep file, `examples/usleep_no_hydra.py` (torch and numpy only), read-only from
its path. Sections 1–7 were written from source:

- `upstream/` is the official U-Time/U-Sleep repository (TensorFlow).
- The preprocessing that U-Time delegates to the external package `psg_utils` (resampling,
  clipping, scaling) was read from the PyPI wheels `psg_utils` 0.1.6 and 0.1.7. The files involved
  are byte-identical in the two versions. The wheels were unzipped in a scratch directory and are
  not part of this folder.
- The AnySleep PyTorch port was read from `../anysleep/upstream/`, which another agent cloned. Its
  checkpoints were opened with `torch.load(..., weights_only=True)`, only to read tensor shapes.
- `braindecode.models.USleep` was read from the `braindecode` 2.0.0 wheel (current at the time of
  writing), with 0.8.1 and 0.7 read to check older signatures. It was not installed.

References use the form `path:line`. Paths are relative to `upstream/` unless they start with
`psg_utils/`, `anysleep:` (meaning `../anysleep/upstream/`) or `braindecode/`. Parameter counts
were computed analytically from the layer shapes. For the AnySleep checkpoints they were also
counted from the stored tensors. Anything not read directly from code or paper is marked
**inferred**.

## 1. Citation and links

| Item | Value |
|---|---|
| Paper | Perslev M., Darkner S., Kempfner L., Nikolic M., Jennum P. J., Igel C. *U-Sleep: resilient high-frequency sleep staging.* npj Digital Medicine 4, 72 (2021) |
| DOI | 10.1038/s41746-021-00440-5 (open access). Saved as `paper.pdf` (12 pages, 1,107,691 bytes) from `https://www.nature.com/articles/s41746-021-00440-5.pdf`. The supplement, which holds Supp. Tables 2–3 with the per-layer topology and hyperparameters, was **not** downloaded. |
| Official repository | https://github.com/perslev/U-Time (TensorFlow ≥ 2.8 + `psg_utils`) |
| Commit cloned | `7fc4cbf79e5454661f1c0d0768886e4dfbe9d42a` (2026-06-24, "Merge pull request #79 … usleep-v2-onnx-export"), package version 1.1.8 (`utime/version.py`) |
| Paper-era code | Tag `usleep-paper-version` (`62a276d`), present in the clone. Its `hparams.yaml` differs from HEAD only in names (the model class was then called `SingleConvUTime`, in `utime/models/sc_utime.py`, and is the same network) and in Keras API spelling (`lr` instead of `learning_rate`). No hyperparameter value differs. |
| License | MIT, Copyright (c) 2018 Mathias Perslev (`LICENSE.txt`) |
| Official pretrained weights | **Not released** in the repository. Available only through the web service https://sleep.ai.ku.dk (non-commercial) and https://yousleep.ai (`README.md:35-40`). |
| PyTorch port with weights | AnySleep, https://github.com/dslaborg/AnySleep, commit `899bfff284f3eb3e7448bc90990f2230757470a2`, MIT. U-Sleep weights: `models/usleep-run{1,2,3}.pth` (three seeds). Paper: Grieger et al., npj Digit. Med. 9, 829 (2026), DOI 10.1038/s41746-026-03406-7. See `../anysleep/README.md`. |
| Third implementation | `braindecode.models.USleep` (BSD-3-Clause), https://braindecode.org. Version 2.0.0 requires Python ≥ 3.11 and torch ≥ 2.6. No pretrained weights. |

```bibtex
@article{perslev2021usleep,
  title   = {U-Sleep: resilient high-frequency sleep staging},
  author  = {Perslev, Mathias and Darkner, Sune and Kempfner, Lykke and Nikolic, Miki and
             Jennum, Poul J{\o}rgen and Igel, Christian},
  journal = {npj Digital Medicine},
  volume  = {4},
  pages   = {72},
  year    = {2021},
  doi     = {10.1038/s41746-021-00440-5}
}
```

## 2. Paper summary

**Problem.** Sleep staging systems are usually tuned and trained for one cohort, one montage and
one sampling rate, and must be retrained for each new clinic. The authors want one ready-to-use
model that works across cohorts and PSG protocols without retraining.

**Method.** U-Sleep is a fully convolutional 1-D U-Net with three parts:

1. An encoder of 12 blocks. Each block is a conv with kernel 9, then ELU, BatchNorm and max-pool 2.
   The filter count grows by about √2 per block.
2. A decoder of 12 blocks. Each block does nearest-neighbour upsampling ×2 and a conv with kernel 2,
   concatenates the encoder skip, then applies a conv with kernel 9. Its output is a dense stage
   map at the input rate (128 Hz).
3. A segment classifier: a 1×1 conv with tanh, mean-pooling over each 30 s window (3840 samples),
   a 1×1 conv with ELU, then a 1×1 conv with softmax.

The model takes **one EEG and one EOG channel** from any standard derivation, resampled to 128 Hz,
and has ≈3.1 M parameters. Because the input channel pair is sampled at random during training, the
model is montage-agnostic. At inference the whole night goes through in one forward pass, and
predictions from all EEG×EOG pairs are combined by majority vote. The pooling width can be shortened
after training to get sub-30 s ("high-frequency") stages.

**Data.** 19,924 PSG records from 15,660 participants, in 16 clinical studies and 21 datasets.
Thirteen datasets were split into train/val/test by subject (or family); eight were used for
testing only, never in training or model selection (Table 1).

**Main results** (Table 2, majority vote, global F1 per dataset). Across the 21 test datasets,
weighted mean F1 per stage is W 0.91, N1 0.53, N2 0.86, N3 0.77, REM 0.90, and the mean F1 is
**0.79 ± 0.03** (range 0.73 SVUH to 0.85 CCSHS/CHAT). On the consensus-scored DOD-H and DOD-O,
U-Sleep matches or exceeds the best of five human experts (mean F1 0.79 ± 0.06 and 0.76 ± 0.10;
Table 3). Majority voting was at least as good as the best single channel pair on 19 of 21 datasets.
Classifying OSA from high-frequency stage-transition triplets reaches F1 ≈ 0.9 at
≥ 768 predictions/min, against 0.60 at 2 per minute (Fig. 4).

**ANPHY reference point (not from this paper).** AnySleep retrained U-Sleep in PyTorch and tested
it **zero-shot** on all 28 ANPHY nights, 3 seeds (AnySleep Supp. Table S5). Results:
W 0.89 (.010), N1 0.47 (.001), N2 0.89 (.005), N3 0.87 (.019), REM 0.87 (.002),
**MF1 0.80 (.004), Cohen's κ 0.79 (.006)**. The run used the whole night at 128 Hz, EEG–EOG pairs
and a hard majority vote. The code and log in `anysleep:examples/anphy/` show 79 EEG × 2 EOG = 158
pairs, not the 83 × 2 = 166 that the caption states. The four channels left out are SO1, SO2, ZY1
and ZY2 (see `../anysleep/README.md` §4).

## 3. Native configuration (official code + paper)

### 3.1 Data and preprocessing

| Setting | Value | Source |
|---|---|---|
| Input channels | 2: one EEG + one EOG (`C = 2`) | paper p. 8 "requires at least two input channels (C = 2), one EEG and one EOG"; `utime/bin/defaults/usleep/hparams.yaml:78` `batch_shape: [64, 35, 3840, 2]` |
| Channel sampling (train) | Per sample, one EEG and one EOG drawn uniformly from those available in the record | paper Methods, Optimization step 3; `utime/bin/defaults/usleep/dataset_configurations/dcsm.yaml:24-27` (`channel_sampling_groups`, one EEG list and one EOG list; every dataset config has one) |
| Channels allowed | Any standard EEG derivation (e.g. F3-M2, C4-M1, O1-M2) and any EOG (E1-M2, E2-M2) | `dcsm.yaml:24-27` |
| Sampling rate | **128 Hz** | `dcsm.yaml:22` `set_sample_rate: 128` (the same in every dataset config); paper p. 9 |
| Resampling method | Polyphase: `scipy.signal.resample_poly(psg, new, old, axis=0)` (`method="poly"` is the default) | paper p. 9 "resampled to 128 Hz using polyphase filtering"; `psg_utils/preprocessing/psg_sampling.py:16-17,20,39-40` |
| Band-pass / notch | **None** | No `filter_settings` or `notch_filter_settings` key in any `utime/bin/defaults/usleep/**` file (grep). `psg_utils/dataset/sleep_study/sleep_study.py:414-420` applies filters only when these keys are set. Paper Fig. 1: "no assumptions about … filtering" |
| Re-referencing | None. The dataset's own derivations are used (e.g. C3-M2). | `dcsm.yaml:24-27` |
| Order of operations | strip → (filter, unused) → **clip** → **resample** → **scale** | `sleep_study.py:404, 425, 432, 436` |
| Strip | Training: `strip_to_match` (trim the PSG and hypnogram to match). Prediction: `trim_psg_trailing` | `dcsm.yaml:37-38`; `utime/bin/predict_one.py:67` |
| Clipping (code) | `clip_noisy_values`, `min_max_times_global_iqr: 20`. For each channel, values are clipped to **±20 × IQR of the raw whole-recording channel**. The interval is centred on 0, not on the median, and is computed at the original sampling rate, before resampling and scaling. | `dcsm.yaml:40-42`; `psg_utils/preprocessing/quality_control_funcs.py:71-115` (clip at line 115) |
| Clipping (paper) | Applied after scaling: "clip any value which has an absolute deviation from the median of more than 20 times the IQR". This is equivalent to the code when the channel median is ≈ 0 (**inferred**: usually true for EEG/EOG). | paper p. 9, Pre-processing |
| Normalization | sklearn `RobustScaler` with default settings (median 0, IQR 25–75 % = 1). One scaler **per channel, per recording**, fitted on that recording itself, test records included. `batch_wise_scaling: false`. | `dcsm.yaml:44-45`; `psg_utils/preprocessing/scaling.py:55-63`; paper p. 9 "global, per-subject and per-channel" |
| Labels | W 0, N1 1, N2 2, N3 3, REM 4. UNKNOWN/MOVEMENT (5) are kept in the input but masked from the loss and from scoring. R&K S3+S4 are merged into N3. | `dcsm.yaml:29-35`; `hparams.yaml:110-112` `ignore_out_of_bounds_classes: true`; paper p. 9 |

### 3.2 Window lengths

| Setting | Value | Source |
|---|---|---|
| Training window | **35 epochs × 30 s = 17.5 min = 134,400 samples** at 128 Hz | paper p. 8–9; `hparams.yaml:78` (35 × 3840) and `hparams.yaml:98` `margin: 17` (2·17+1 = 35) |
| Inference window | **The whole night in one forward pass.** The model is rebuilt for each study with `n_periods = study.n_periods`. | paper p. 8 "the whole PSG is input to U-Sleep"; `utime/bin/predict_one.py:505-510`; `utime/bin/predict.py:412-413` (`--one_shot`) |
| Minimum length | No hard limit (convs use same padding, and odd lengths are padded). The paper calls ≥ 4096 samples "ideal" and says that < 17.5 min "may reduce performance". | paper p. 8 |
| Channel fusion at inference | Every EEG×EOG pair is predicted. The **softmax scores are summed** and the argmax is taken (a soft vote). | paper "Input channel majority voting"; `utime/bin/predict.py:255-260, 263-270` (`no_argmax=True` before summing); `utime/bin/predict_one.py:180` |

### 3.3 Model (`utime/models/usleep.py`, class `USleep`, line 133)

| Hyperparameter | Value | Source |
|---|---|---|
| depth | 12 encoder blocks + bottom + 12 decoder blocks | `hparams.yaml:71`; `usleep.py:258-283, 301-325` |
| kernel_size | 9 (encoder, bottom, decoder conv2); 2 (decoder conv1 after upsampling) | `hparams.yaml:72`; `usleep.py:260, 276, 307, 319` |
| dilation | 1 | `hparams.yaml:73` |
| activation | ELU in every conv | `hparams.yaml:70` |
| init_filters | 5 (constructor default; not overridden) | `usleep.py:152` |
| complexity_factor | 1.67. The code multiplies filter counts by **√1.67 = 1.2923**. The constructor default is 2, but the hparams set 1.67. | `hparams.yaml:75`; `usleep.py:153, 201` |
| Filter growth | Encoder: `filters = int(filters·√2)` per block, conv width `int(filters·cf)`. Decoder: `filters = ceil(filters/√2)`. | `usleep.py:260, 272, 302, 307` |
| Resulting widths | Encoder 6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214; bottom 302; decoder 214 … 6 (mirror) | computed from the above |
| Pooling | MaxPool 2, stride 2. An odd length is first zero-padded by 1 at the **start**. | `usleep.py:104-110, 267-268` |
| Upsampling | Nearest ×2, then conv k=2 (`same`), ELU, BN. **Centre-crop** to the skip length. Concat `[skip, up]`, then conv k=9, ELU, BN. | `usleep.py:306-324, 123-130` |
| Block order | Conv → ELU → BatchNorm (Keras defaults: momentum 0.99, eps 1e-3) | `usleep.py:260-266` |
| Dense head | 1×1 conv to `int(5·√1.67) = 6` channels, **tanh** | `usleep.py:148, 327-342` |
| Segment classifier | AvgPool(3840) → 1×1 conv (5, ELU) → 1×1 conv (5, softmax). `transition_window: 1`. | `hparams.yaml:74`; `usleep.py:363-380` |
| Init | Glorot-uniform kernels, zero biases | `usleep.py:154-155` |
| Regularization | None (`l2_reg: Null`). Paper: "regularization unnecessary". | `hparams.yaml:77`; paper p. 9 |
| Parameters | **3,114,337 trainable** + 5,014 BN moving statistics = 3,119,351 in total (computed). Paper: "≈3.1·10⁶". | paper p. 8 |
| Receptive field | Paper: ≈9.6 min at the last encoder conv; outputs are sensitive to ±6.75 min of input. Standard RF arithmetic over the encoder gives 69,624 samples = **543.9 s ≈ 9.1 min** (computed; the paper's 9.6 min likely uses a slightly different count, **inferred**). | paper p. 9 |

### 3.4 Training

| Setting | Value | Source |
|---|---|---|
| Optimizer | Adam, **AMSGrad**, **lr 1.0e-7**, β 0.9/0.999, ε 1e-8, decay 0, constant LR | `hparams.yaml:118-119`; paper p. 9 "learning rate η = 10⁻⁷" |
| Batch | **64** sequences × 35 epochs = 2240 labels per step | `hparams.yaml:115`; paper p. 9 |
| Loss | Unweighted sparse categorical cross-entropy; out-of-range labels ignored | `hparams.yaml:101, 112`; paper p. 9 |
| "Epoch" | 10⁶ sampled 30 s segments: `--max_train_samples_per_epoch 1000000` / 35 per sequence = 28,571 sequences ≈ 447 steps of 64 (paper says 443) | `README.md:282,289`; `utime/utils/scriptutils/train.py:255-261`; paper p. 9 |
| Max epochs | 12,000 | `hparams.yaml:116` |
| Early stopping | **Code**: `val_dice`, patience **200**, mode max. **Paper**: 100 epochs without validation-loss improvement. The two disagree. | `hparams.yaml:9-14, 122`; paper p. 9 |
| Class balancing | Stratified sampling, `balanced_sampling: True`. (1) Draw a class uniformly. (2) Draw a random epoch of that class from a random record that has it. (3) Shift it to a random position in the 35-epoch window (±17 epochs). The other 34 labels are left unbalanced. | `hparams.yaml:95`; `utime/sequences/balanced_random_batch_sequence.py:80, 127-153`; paper p. 9 |
| Dataset sampling | `P = (1-α)·P_size + α·P_uniform`, with α = 0.5 | `utime/sequences/multi_sequence.py:44, 69-77`. The paper's formula has a typo (`αP1(D)·(1−α)P2(D)`, a product). |
| Subject sampling | Uniform within the chosen dataset | paper p. 9 step 2 |
| Augmentation | `RegionalErase`: with p = 0.1 per sample, a region (fraction log-uniform in [0.001, 0.33]) is replaced by Gaussian noise. `ChannelDropout`: with p = 0.1 per sample, `drop_fraction 0.5` of 2 channels = 1 channel is replaced by noise. The noise has mean = the segment mean and **std 0.01**. The paper states σ² = 0.01 (std 0.1) and a fraction range of [0.001, 0.3]. | `hparams.yaml:80-91`; `utime/augmentation/augmenters.py:377-400, 429-450`; paper p. 9 |
| Augmentation weight | `aug_weight 0.5` would scale the loss weight of augmented samples. But `process_batch` does not return weights (`utime/sequences/base_sequence.py:403`), so it has no effect (**inferred**). | `augmenters.py:20, 126` |
| Hardware | One Titan X (12 GiB). The same model was trained once, without tuning. | paper p. 8, 10 |

### 3.5 Evaluation (paper)

- Score the full PSG. Ignore segments whose label is not one of the 5 stages.
- Fuse all EEG×EOG pairs with the soft majority vote.
- F1 per class from non-normalized confusion matrices; MF1 is the unweighted mean of the 5 classes.
- Table 2 reports a global F1 over all subjects of a dataset. Table 3 reports per-subject mean ± sd.
- Consensus datasets (DOD) are scored against an N−1 consensus with the Dreem evaluation code.
- Cohen's κ is **not** reported in the paper. AnySleep reports it.

## 4. The three implementations compared

| | Official U-Time (TF/Keras) | AnySleep port (PyTorch) | `braindecode.models.USleep` 2.0.0 |
|---|---|---|---|
| Class | `utime.models.usleep.USleep` | `anysleep:base/model/usleep.py` `USleep` (needs the Hydra `Config` singleton); the standalone copy is `anysleep:examples/usleep_no_hydra.py` `USleep` (imports only torch/numpy) | `braindecode.models.USleep` |
| Constructor | `USleep(n_classes, batch_shape, depth=12, dilation=1, activation="elu", dense_classifier_activation="tanh", kernel_size=9, transition_window=1, padding="same", init_filters=5, complexity_factor=2, l2_reg=None, data_per_prediction=None)`, with hparams overriding to cf 1.67 | `USleep(path=None, depth=12, complexity=1.2923, scale=1.4142, sleep_stage_frequency=1)` (`usleep_no_hydra.py:69-77`). Input channels are hard-coded to 2 (`:80`), the rate to 128 Hz and the epoch to 30 s (`:217-218`), and the kernel to 9 (`connector` and block defaults). | `USleep(n_chans, sfreq, depth=12, n_time_filters=5, complexity_factor=1.67, with_skip_connection=True, n_outputs=5, input_window_seconds, time_conv_size_s=9/128, ensure_odd_conv_size=False, activation=nn.ELU, chs_info=None, n_times=None)` (`braindecode/models/usleep.py:174-189`) |
| Input layout | `[B, n_periods, 3840, 2]`, reshaped internally to `[B, T, 1, C]` | **`[B, time, 2]`** (time-major; channel 0 EEG, channel 1 EOG), transposed internally to `[B, 2, time]` (`usleep_no_hydra.py:117-120`) | `[B, C, T]`, or `[B, S, C, T_epoch]`, which is flattened to `[B, C, S·T_epoch]` (`:306-308`) |
| Output | `[B, n_periods, 5]` **softmax** | `[B, n_epochs, 5]` **logits** (softmax commented out; `usleep_base_components.py:144`) | `[B, 5, S]` **logits**. `[B, 5]` if S = 1. |
| Widths / params | 6 … 214, 302; 3,114,337 | Identical widths (`utilities.py:34-42` reproduces the int/√2 chain; `scale=1.4142` and `complexity=1.2923` give the same integers). **3,114,337**, verified by counting the tensors of all three checkpoints (265 tensors each). | Identical widths (`:219-224`); 3,114,337 (computed) |
| Decoder widths | `ceil(f/√2)` chain | Mirror of the encoder list. The integers are the same here (computed). | Mirror of the encoder list (same) |
| Odd-length pooling | Pad 1 at start; the skip is the padded tensor | Same (`usleep_base_components.py:47-49`) | `ConstantPad1d(1)` on both ends before the pool, so the windows are the same as the original; the skip is the **unpadded** tensor (`:364-372`) |
| Skip merge | Centre-crop the upsampled map; concat `[skip, up]` | Same (`usleep_base_components.py:98-107`) | **Left-aligned** crop of both to the shorter length; concat `[up, skip]` (`:422-425, 430-438`) |
| Dense head + classifier | conv1×1 (6, tanh) → avgpool → conv(5, ELU) → conv(5, softmax) | conv1×1 (6→6, tanh) → avgpool → conv(6→5) → ELU → conv(5→5) (`usleep_base_components.py:137-145`) | Same structure, no softmax (`:268-298`) |
| BatchNorm | Keras defaults (momentum 0.99, eps 1e-3) | PyTorch defaults (momentum 0.1, eps 1e-5) | PyTorch defaults. **BN is switched to eval when batch size = 1** (`braindecode/models/util.py:124-145`). |
| Init | Glorot-uniform | Kaiming-normal (`anysleep:base/model/base_model.py:16-40`) | PyTorch default |
| Kernel at other rates | Fixed in samples | Fixed in samples (9) | `round(time_conv_size_s·sfreq)`; it must be odd, or `ensure_odd_conv_size=True` adds 1 (`:207-217`) |
| Softmax flag history | | | 0.7: `apply_softmax=False`, `in_chans`, `n_classes`, `input_size_s`. 0.8: `add_log_softmax`, with the old names deprecated. 2.0.0: no softmax option, always logits. |
| Pretrained weights | Not public | `usleep-run{1,2,3}.pth`, ~12.5 MB each, key `state_dict` | None |

**Verdict.**

- The AnySleep port **is the official architecture**: same widths, kernels, padding side,
  centre-crop, concat order, head and parameter count. It differs only in BN
  momentum/eps and initialization, which do not matter for inference with its own weights.
- braindecode reproduces widths and parameter count. It differs from the original at odd
  lengths (skip alignment) and in concat order. That is irrelevant when training from scratch,
  but its weights could not be swapped with the other two without remapping (**inferred**).

**How the AnySleep U-Sleep weights were trained.** This is AnySleep's own pipeline, not Perslev's:

- Data: the original U-Sleep split, 13 training datasets (`anysleep:config/usleep_split.yaml`).
- Optimizer: Adam with **AMSGrad, lr 1e-5** (100× the official 1e-7), weight decay 0, constant LR
  (`anysleep:config/exp001/usleep_config.yaml:55-78`).
- Batch: **64 × 35-epoch windows**. Each epoch is 28,571 samples ≈ 447 steps (`usleep_config.yaml:13, 20-21`).
- Stopping: at most 10,000 epochs, early stop on **validation macro-F1**, patience 100
  (`usleep_config.yaml:72-74`; `anysleep:base/training/clas_trainer.py:107-150`).
- Sampling: stratified. A class is drawn uniformly, then a dataset by
  `0.5·uniform + 0.5·size`, a subject, an epoch of that class placed at a random position in the
  window, and one random EEG and one random EOG channel
  (`anysleep:base/data/usleep_train_dataset.py:77-117, 119-156, 219-232`).
- Loss: plain `CrossEntropyLoss` with artefact label 9 masked (`clas_trainer.py:202-205`).
- **No augmentation** (AnySleep paper §4.3).
- Preprocessing: done by the external `dslaborg/Sleep-Datasets` repository. For EDF inference
  the example scripts do `resample_poly` to 128 Hz → clip ±20·IQR about 0 → per-channel
  `RobustScaler` on the whole recording (`anysleep:examples/anphy/predict_anphy_usleep.py:39-96`).
- Reported weighted test MF1 is 0.791, against 0.79 for the original U-Sleep (AnySleep paper §4.3).

## 5. Which implementation to use

### (a) Zero-shot, native configuration, pretrained weights → AnySleep port

```python
import sys, torch
sys.path.insert(0, "baselines/anysleep/upstream/examples")
from usleep_no_hydra import USleep

model = USleep(path="baselines/anysleep/upstream/models/usleep-run1.pth",
               sleep_stage_frequency=1).eval()
x = torch.as_tensor(pairs, dtype=torch.float32)
logits = model(x)
```

- `pairs` has shape `[n_pairs, n_epochs * 3840, 2]`, 128 Hz, channel 0 EEG and channel 1 EOG.
- `logits` has shape `[n_pairs, n_epochs, 5]`, classes W, N1, N2, N3, REM.
- Repeat with `usleep-run2.pth` and `usleep-run3.pth`, and report the mean ± sd over the three
  runs, as AnySleep did.
- `path` is opened relative to the working directory (`usleep_no_hydra.py:138-143`) with
  `weights_only=True`.
- Make the input length a multiple of 3840. The trailing partial epoch is dropped by `AvgPool1d`
  (**inferred** from floor pooling).

Native preprocessing, as AnySleep ran it on ANPHY:

1. Raw EDF channels: EEG with the recording reference, EOG1/EOG2. No band-pass, no notch, no CAR.
2. `scipy.signal.resample_poly(x, 128, 1000)`, which reduces to up 16 / down 125.
3. Clip each channel to ±20·IQR of the whole night.
4. `RobustScaler` per channel per night.
5. Feed the whole night.
6. Fuse every EEG×EOG pair. The paper uses a soft vote (sum of softmax). AnySleep's ANPHY script
   uses a hard vote with random tie-break (`predict_anphy_usleep.py:99-133, 392-396`). Pick one
   and state it.

Reasons:

- The official TF weights are not downloadable.
- This port is architecturally identical to the official model and checkpoint-verified at 3,114,337 parameters.
- It needs only torch and numpy. Preprocessing adds scipy, scikit-learn and mne, all already in our venv.
- It has a published ANPHY reference (MF1 0.80, κ 0.79) to check our numbers against.

Caveats:

- The weights come from a **re-training** (lr 1e-5, no augmentation), not from Perslev's run.
- The native setting is **non-causal**: whole-night context and whole-night scaling statistics.
- It uses **EOG**, which our models do not.
- The EEG reference differs from the mastoid derivations used in training (ANPHY EDF channels
  carry the recording reference, **inferred** from the channel names).

Score it on our fold's test subjects (for comparison with our models) and also on all 28 nights
(for comparison with AnySleep Table S5). The weights never saw ANPHY, so all nights are valid
test data.

### (b) Training from scratch → AnySleep port's blocks (primary), braindecode (alternative)

**Primary: wrap the AnySleep blocks** (`USleepEncoderBlock`, `USleepDecoderBlock`,
`SegmentClassifier`, `calc_filters`) in a small module of our own whose constructor takes
`n_channels`, `sfreq`, `epoch_seconds` and `kernel_size` explicitly. Reasons:

- It is the faithful architecture, including centre-crop, concat order and the padded skip.
- It shares one code path with (a).
- It adds **no new dependency** (AGENTS.md).
- The constants hard-coded in the port (2 channels, 128 Hz, kernel 9 in the connector) are exactly
  the ones we must change. With no default values, this also follows the repository convention.

Not implemented, per the task.

**Alternative: braindecode** (when a maintained package is preferred):

```python
from braindecode.models import USleep
model = USleep(n_chans=83, sfreq=1000, n_outputs=5, input_window_seconds=30,
               depth=12, n_time_filters=5, complexity_factor=1.67,
               with_skip_connection=True, time_conv_size_s=9 / 1000)
```

- For a kernel of 9 samples use `time_conv_size_s=9 / 1000`. For the same 70 ms as native use
  `time_conv_size_s=9 / 128, ensure_odd_conv_size=True`, which gives a kernel of 71.
- Input `[B, 83, 600000]` or `[B, 20, 83, 30000]`; output logits `[B, 5, 20]`.
- Costs: it pulls in skorch, torchaudio (which must match our torch 2.13 build), einops, pydantic,
  mne_bids, wfdb, h5py, torchinfo and docstring_inheritance. Our venv already has Python 3.11.15,
  torch 2.13 and mne 1.13.2, so those meet its pins.
- Its skip alignment at odd lengths differs from the original.
- It silently stops BN updates at batch size 1.

## 6. Our configuration (assessment only; nothing implemented)

Our protocol: 83 EEG channels at 1000 Hz, never resampled; 0.5–45 Hz band-pass + 60 Hz notch +
common average reference; `RobustScaler` fitted on training subjects, clipped to ±20; EEG only.
One item is a run of 20 consecutive 30 s epochs (600,000 samples) that ends at the scored epoch,
and the prediction must use only that epoch and the 19 before it.

1. **Input channels 2 → 83.** Only the first conv changes, from 2→6 to 83→6 (+4,374 weights), for
   **3,118,711** parameters. Every later width is unchanged. The first layer compresses
   83 electrodes into 6 feature maps, a hard spatial bottleneck (**inferred** to limit what U-Sleep
   can use from the montage). Widening it is a separate experiment, not part of "U-Sleep in our
   configuration".
2. **Segment pooling 3840 → 30,000 samples** (30 s × 1000 Hz). 600,000 / 30,000 = 20 outputs,
   which divides exactly.
3. **Divisibility by 2^depth is not required.** 600,000 = 2⁶ · 9375. Level lengths are
   600000, 300000, 150000, 75000, 37500, 18750, **9375**, 4688, 2344, 1172, 586, **293**, then a
   bottom of 147. The two odd levels are handled by pad-at-start and the centre-crop. The native
   134,400 = 2⁸ · 525 also has odd levels (525, 263), so this mechanism is part of the native
   model, not something new. Padding the input to a multiple of 4096 would be an optional extra.
4. **Receptive field: the main problem.** The kernel is fixed in samples, so at 1000 Hz each
   kernel and the whole RF shrink in seconds by 1000/128 = 7.8×. The encoder RF is 69,624 samples
   = **69.6 s (≈ 2.3 epochs)**, against 543.9 s at 128 Hz. A 9-sample kernel spans 9 ms instead
   of 70 ms. Options (computed):

   | Variant | Kernel | Depth | Encoder RF at 1 kHz | Params (83 in) | Note |
   |---|---|---|---|---|---|
   | Native widths, kernel in samples | 9 | 12 | 69.6 s | 3.12 M | Same parameter count; loses the multi-minute context U-Sleep is built around |
   | Kernel in seconds (braindecode `time_conv_size_s=9/128`) | 71 (70 → odd) | 12 | 577.5 s ≈ 9.6 min | **22.75 M** | Native context, 7.3× the parameters |
   | Deeper | 9 | 15 | 557.0 s | 24.4 M | Bottom length ≈ 18; also 7.8× the parameters |
   | Decimating stem (fixed or strided) | 9 | 12 | ≈ native | ≈ 3.1 M | **Violates the no-resampling invariant**, even when learned (**inferred**) |

   Our window is 10 min (20 epochs), shorter than the native 17.5 min training window. Recommended:
   run the 3.12 M variant as "U-Sleep, our configuration" so the parameter count matches, and the
   kernel-71 variant as a sensitivity check. Report both parameter counts.
5. **Causality.** U-Sleep is bidirectional: same-padded convs plus a U-Net decoder, and the paper
   shows ±6.75 min sensitivity. Feed the 20-epoch run that ends at the scored epoch and **score
   only output index 19**. Outputs 0–18 use future context inside the run and must not be scored.
   This is causal by construction, at 20× redundant compute at test time.

   Training choice (**inferred** trade-off):
   - Loss on index 19 only: matches test-time conditions exactly, but uses 1/20 of the labels.
   - Loss on all 20 outputs: 20× the supervision, but most targets then see future context the
     test item will not have.

   The right edge of the last epoch is zero-padded in both cases. In either case the
   whole-recording scaler and the whole-night input of the native setting are replaced by our
   train-fitted scaler and 20-epoch runs.
6. **Preprocessing.** Our cache replaces the native pipeline completely: no resampling, band-pass +
   notch + CAR, train-subject scaler. This only makes sense for training from scratch. The
   pretrained weights expect the native pipeline in (a), so do not mix the two.
7. **Training recipe.** The native lr of 1e-7 was set for ~15k training nights and millions of
   steps. For ~20 training nights, the AnySleep recipe (AMSGrad, lr 1e-5, stratified class
   sampling, early stop on validation MF1) or our own protocol's optimizer settings are the
   realistic choices (**inferred**). Whichever is used, state it.
8. **Memory** (estimated: fp32, training mode, counting the activations autograd keeps; **inferred**
   accuracy ±50 %):

   | Case | Input | Saved activations | Note |
   |---|---|---|---|
   | Native training, 64 × 35 × 3840 × 2 | 66 MiB | ≈ 7.1 GiB | Consistent with the paper's 12 GiB Titan X |
   | Ours, batch 1, 83 × 600,000 | 190 MiB | **≈ 0.7 GiB** | Weights + Adam state: 3.1 M → ~50 MB; 22.75 M → ~360 MB. Total peak ≈ 1–1.5 GiB |
   | Ours, batch 64 | 11.9 GiB | ≈ 43 GiB | ≈ 55–60 GiB; fits one A100 80 GB, tight |
   | Native whole night (8 h, 128 Hz, 2 ch), batch 1 | 28 MiB | ≈ 3 GiB (training mode) | Inference under `no_grad` is far smaller |

   Activation memory does not depend on kernel size. The kernel-71 variant mainly costs compute,
   about 8× the conv FLOPs (**inferred**).

## 7. Files

| Path | Content |
|---|---|
| `README.md` | This document |
| `paper.pdf` | Perslev et al. 2021, npj Digit. Med. 4:72, open access from nature.com (1,107,691 bytes) |
| `upstream/` | `git clone https://github.com/perslev/U-Time` at `7fc4cbf79e5454661f1c0d0768886e4dfbe9d42a` (MIT). Read only, never installed or run. The relevant files: `utime/models/usleep.py`, `utime/bin/defaults/usleep/hparams.yaml`, `utime/bin/defaults/usleep/dataset_configurations/*.yaml`, `utime/sequences/balanced_random_batch_sequence.py`, `utime/sequences/multi_sequence.py`, `utime/augmentation/augmenters.py`, `utime/bin/predict.py`, `utime/bin/predict_one.py`. |
| `../anysleep/upstream/` | AnySleep clone (another agent's folder): the PyTorch U-Sleep and the `models/usleep-run{1,2,3}.pth` weights |
| `__init__.py`, `model.py`, `runner.py` | The runner (§8) |
| `native.yaml`, `ours.yaml` | The two configurations (§8) |

## 8. How to run / implementation decisions

```bash
.venv/bin/python -m baselines.run usleep native      # zero-shot, released weights
.venv/bin/python -m baselines.run usleep ours        # from scratch, our protocol
# add --quick for a smoke test
```

Outputs go to `baselines/results/usleep/<config>[_quick]/`. The native night caches go to
`baselines/cache/usleep/<config>[_quick]/`.

### Code

- `model.py` loads `../anysleep/upstream/examples/usleep_no_hydra.py` read-only from its path
  (`importlib`, nothing added to `sys.path`, nothing copied).
- `pretrained_usleep` returns their `USleep` with a released checkpoint.
- `USleepNetwork` assembles their `USleepEncoderBlock`, `USleepDecoderBlock`, `SegmentClassifier`
  and `calc_filters`. It exposes only what the port hard-codes: input channels, kernel size (encoder,
  connector, decoder) and the segment classifier's pooling width. The pooling layer of their
  `SegmentClassifier` is replaced by `AvgPool1d(pool_samples)`.
- **Equivalence check (done):** `USleepNetwork(2, 12, 9, 1.2923, 1.4142, 3840)` loads
  `usleep-run1.pth` with `strict=True`. On random 35- and 37-epoch inputs its logits equal those of
  their `USleep` with max |diff| = 0.0. This is the official architecture, 3,114,337 parameters.
- Initialisation for training from scratch reproduces AnySleep's `_weights_init`: Kaiming-normal
  fan-out convs, BN 1/0. Their `base_model.py` imports Hydra, so it is reproduced, not imported.
- No braindecode.

### native: zero-shot with the released (unofficial) weights

- `usleep-run{1,2,3}.pth` are the AnySleep authors' PyTorch retrain of U-Sleep. They are
  **unofficial**; Perslev et al.'s weights are not released.
- Nothing is trained on ANPHY. **No native training from scratch:** the official recipe (lr 1e-7,
  up to 12,000 epochs of ~447 steps, ~15k nights) is infeasible on 18 nights and would not be "native".
- Pairs: 79 EEG (our 83 montage electrodes minus SO1, SO2, ZY1, ZY2, as in AnySleep's ANPHY
  script) × EOG1/EOG2 = **158 pairs**.
- Per channel, per night: `resample_poly` 1000 → 128 Hz → clip to ±20·IQR (centred on 0, as
  `clip_noisy_values`) → `RobustScaler` fitted on that night. No filter, no re-reference.
- This is AnySleep's order (resample, then clip). Official U-Time clips before resampling (§3.1);
  the difference is negligible (**inferred**).
- Alignment: the night is cropped to start at the first hypnogram row of `night_epochs` (onset 0
  on all test nights, rows contiguous every 30 s) and to hold exactly one 3840-sample epoch per row.
  Output k is therefore row k. Unscored rows stay in the input, as in U-Sleep, and are not saved.
- Each pair gets the **whole night** in one pass. Pairs are combined by the **official rule, the
  sum of softmax** (`utime/bin/predict.py:255-270`), into `usleep-runK_soft/`. AnySleep's hard
  majority vote (random tie-break, seed `tie_seed`) goes into `usleep-runK_hard/` as a secondary row.
- Each folder is scored by `score_run` on the 5 test nights. `metrics.json` has `test` (mean over
  the 3 checkpoints, soft rule), `test_sd`, `summary.soft` / `summary.hard` (mean, sd, values),
  every run, and the parameter count (3,114,337).
- The native setting is non-causal (whole night, whole-night scaling statistics) and uses EOG.
  Report it as such.

### ours: from scratch on our protocol

- Our data, our stage-2 trainer and our scoring, through `baselines.common.ours.run_ours`.
- Overrides: `{finetune: {freeze_epochs: 0}}`, as in Exp 15a.
- Architecture: the released one with only the forced changes. First conv 83→6 instead of 2→6;
  segment pool 30,000 samples (30 s at 1 kHz). Kernel 9 samples, depth 12, complexity 1.2923,
  scale 1.4142. All are YAML values, so the receptive-field sensitivity runs (§6.4) are config
  changes. **3,118,711 parameters.**
- The input `[B, 20, 83, 30000]` is concatenated in time to `[B, 83, 600000]`. The output is
  `[B, 20, 5]` logits.
- **Causality** comes from the protocol: one test item per scored epoch, ending at it; only its
  position is labelled. Training uses labels on all clean positions, as the shared trainer does.
- `classifier` (the optimiser's head group: lr 1e-3, backbone 1e-4 via `backbone_lr_scale`) is
  U-Sleep's whole segment classifier (1×1 conv/tanh, pooling, 1×1 conv/ELU, 1×1 conv; 107 params).
- Seed: `torch.manual_seed(42)` before building the network.

### Smoke tests (done 2026-11-03, outputs deleted afterwards)

- `native --quick` (1 night EPCTL06, 4 pairs, run1 only): finished in 28 s wall. Preprocessing of
  4 channels took 17 s; prediction 5.7 s. 886 scored epochs; κ 0.808 / MF1 0.804 (soft), κ 0.795 (hard).
- Full native estimate (**inferred**): ~6 min preprocessing per night for 81 channels, done once
  and cached, ~1.2 GB per night at 128 Hz.
- `ours --quick` (2 train / 1 val / 1 test nights, 1 epoch, 48 test items): finished in 3 min 54 s
  wall, 11 GB peak RAM. Training epoch: 51 steps of batch 2 in 41 s, data-loader bound. Scores are
  meaningless after 1 epoch (κ ≈ 0).
- GPU, measured separately with `torch.cuda.max_memory_allocated` for one training step, fp32,
  AdamW, input `[B, 20, 83, 30000]`:

  | Batch | Peak GPU memory | Train step | Eval forward |
  |---|---|---|---|
  | 1 | 0.86 GiB | 29 ms | 10 ms |
  | 2 (`finetune.batch_size`) | 1.69 GiB | 53 ms | 17 ms |

  Compute is ~50 ms per step. Wall time per step in the trainer (~0.25–0.8 s) is set by loading the
  83 × 600,000 runs.
