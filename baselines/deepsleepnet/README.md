# DeepSleepNet baseline

Setup notes for running DeepSleepNet as a baseline on ANPHY-Sleep. This folder holds only the
reference material and the adaptation plan. Nothing upstream has been executed.

Line references `file:line` point into `upstream/` at the commit below. "Paper §X" refers to
`paper.pdf` (arXiv v2). Anything marked **inferred** comes from reading the code or from
standard library behaviour; the authors do not state it.

## 1. Citation and links

| | |
|---|---|
| Paper | A. Supratak, H. Dong, C. Wu, Y. Guo, "DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG", *IEEE TNSRE* 25(11):1998–2008, 2017, doi:10.1109/TNSRE.2017.2721116 |
| arXiv | https://arxiv.org/abs/1703.04046 (v2, 3 Aug 2017), saved as `paper.pdf` |
| Official code | https://github.com/akaraspt/deepsleepnet, cloned into `upstream/` at commit `eb72f9b27af54e6caa1f1bd89a285c58ebfe7779` (2024-07-19, "Merge pull request #49") |
| Code license | Apache License 2.0 (`upstream/LICENSE`, "Copyright 2017 Akara Supratak and Hao Dong"). The README adds "For academic and non-commercial use only" |
| Framework | TensorFlow 1.x (`tf.compat.v1`, `tf.contrib.legacy_seq2seq`), Python 3.5, tensorflow-gpu 1.15.2, mne 0.20 (upstream README) |
| PyTorch implementation | **No faithful port exists** (§5). The closest maintained one is `braindecode.models.DeepSleepNet` (braindecode v2.0.0, tag commit `64888d7debfaedb9367593c6a585db779719a420`, BSD-3-Clause). It covers the epoch encoder only. We therefore wrote our own port, `model.py`, to the specification in §5.3 (see §8). No port is cloned into `port/` |

```bibtex
@article{Supratak2017,
  title   = {DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG},
  author  = {Supratak, Akara and Dong, Hao and Wu, Chao and Guo, Yike},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year    = {2017}, volume = {25}, number = {11}, pages = {1998--2008},
  doi     = {10.1109/TNSRE.2017.2721116}
}
```

## 2. Paper summary

- **Problem.** Five-class sleep staging (W, N1, N2, N3, REM) of 30 s epochs from one raw EEG
  channel, with no hand-engineered features.
- **Method.** The model has two parts.
  - *Representation learning:* two CNNs read each epoch. One has a small first filter (Fs/2,
    for temporal precision) and the other a large one (4·Fs, for frequency precision). Their
    outputs are concatenated (paper §II-A, Fig. 1).
  - *Sequence residual learning:* a 2-layer bidirectional LSTM (512 units per direction, with
    peephole connections) runs over a sequence of epoch features. A shortcut FC layer
    (1024, BN, ReLU) adds the CNN features to the LSTM output (paper §II-B, eq. 4–6).
  - *Training:* two steps (paper §III, Alg. 1). First the CNNs are pretrained with a
    temporary softmax layer on a class-balanced (oversampled) training set. Then the whole
    model is fine-tuned on sequential, non-oversampled data, with lr 1e-6 for the CNNs and
    1e-4 for the rest.
- **Data.** There are two datasets.
  - *MASS SS3:* 62 subjects, AASM scoring, F4-EOG(Left) at 256 Hz, 31-fold CV.
  - *Sleep-EDF (2013 SC, 20 subjects / 39 nights):* R&K scoring with N3+N4 merged, Fpz-Cz
    and Pz-Oz at 100 Hz, 20-fold CV.
  - In both, the folds are subject-independent, and test predictions are pooled over all
    folds before scoring (paper §IV-A/B).
- **Main results** (paper Table IV):

| Dataset | Channel | Test epochs | ACC | MF1 | κ | F1 W / N1 / N2 / N3 / REM |
|---|---|---|---|---|---|---|
| Sleep-EDF | Fpz-Cz | 41,950 | 82.0 | 76.9 | 0.76 | 84.7 / 46.6 / 85.9 / 84.8 / 82.4 |
| Sleep-EDF | Pz-Oz | 41,950 | 79.8 | 73.1 | 0.72 | 88.1 / 37.0 / 82.7 / 77.3 / 80.3 |
| MASS | F4-EOG(L) | 58,600 | 86.2 | 81.7 | 0.80 | 87.3 / 59.8 / 90.3 / 81.5 / 89.3 |

- **Without sequence residual learning** (MASS, CNN pre-model only, Table V): MF1 drops to
  76.6 and ACC to 81.5. These numbers are recomputed from the confusion matrix (**inferred**;
  the paper gives only per-class F1).
- **Data requirement.** The authors report that performance drops when the training set has
  fewer than about 54,000 epochs (paper §V). ANPHY has 23,407 scored epochs in total, so a
  training split of about 18–19k epochs is well below that.
- **External reference (not ours).** MTSSRL-MD (ICLR 2026 submission) reports DeepSleepNet on
  ANPHY at BACC 78.90, MF1 77.34, κ 75.90. That setting used 11 EEG channels at 100 Hz,
  0.3–40 Hz, and an 80/20 subject split. It is not the native single-channel setting and not
  our protocol, so it serves only as a reference point.

## 3. Native configuration (exact, with sources)

### 3.1 Data preparation (Sleep-EDF, `prepare_physionet.py`)

| Item | Native value | Source |
|---|---|---|
| Channel | `EEG Fpz-Cz`, a bipolar channel stored as such in the Sleep-EDF EDF | `prepare_physionet.py:66`, `README.md` ("--select_ch 'EEG Fpz-Cz'"); paper §IV-A |
| Sampling rate | 100 Hz, as recorded. The script never resamples, and the model hard-codes `input_dims = 30*100 = 3000` | `prepare_physionet.py:93`; `train.py:40,60`. `deepsleep/sleep_stage.py:20` `SAMPLING_RATE = 256` is unused (**inferred**: a leftover from MASS) |
| Filtering | **None** in software ("without any further pre-processing") | paper §IV-A; no filter call in `prepare_physionet.py` |
| Units | µV (**inferred**: `raw.to_data_frame()` with MNE's default `scalings`, eeg = 1e6) | `prepare_physionet.py:94` |
| Normalization | **None**. The loaders only squeeze and cast to float32 | `deepsleep/data_loader.py:58-68,209-217` |
| Epochs | 30 s, so 3000 samples, from hypnogram annotations | `prepare_physionet.py:57,127-133,182` |
| Labels | W 0, S1→N1 1, S2→N2 2, **S3 and S4→N3** 3, R→REM 4 | `prepare_physionet.py:46-55`; paper §IV-A |
| MOVE / UNK | `Sleep stage ?` and `Movement time` are mapped to UNKNOWN. Their samples are cut out of the signal and the rest is concatenated | `prepare_physionet.py:53-54,126-158` |
| Wake trimming | Keep only **30 min (60 epochs)** of W before the first non-W epoch and after the last one (`w_edge_mins = 30`, `±w_edge_mins*2` epochs) | `prepare_physionet.py:187-198`; paper §IV-A |
| Tail trimming | Signal past the last full labelled epoch is dropped | `prepare_physionet.py:161-171` |

### 3.2 Architecture (`deepsleep/model.py`, `deepsleep/nn.py`)

Each conv block is conv (no bias), then BN, then ReLU. All conv and pool layers use TF
`padding="SAME"`, so `out = ceil(in/stride)` (`nn.py:49-88`). Shapes below are for 100 Hz and
one channel, with input `[N, 3000]`.

| Layer | CNN-1 (small filter) | CNN-2 (large filter) | Source |
|---|---|---|---|
| conv1 | k = **50 = Fs/2**, 64 filters, stride **6 ≈ Fs/16** → 500 | k = **400 = 4·Fs**, 64 filters, stride **50 = Fs/2** → 60 | `model.py:75,116`; paper §II-C, Fig. 1 |
| (MASS 256 Hz, commented out) | k 128, stride 16 | k 1024, stride 128 | `model.py:74,115`, which confirms the Fs scaling |
| pool1 | max 8 / 8 → 63 | max 4 / 4 → 15 | `model.py:79,120` |
| dropout | 0.5 | 0.5 | `model.py:84-91,125-132` |
| conv2–4 | 3 × (k 8, 128 filters, stride 1) → 63 | 3 × (k 6, 128 filters, stride 1) → 15 | `model.py:94-96,135-137` |
| pool2 | max 4 / 4 → 16 | max 2 / 2 → 8 | `model.py:100,141` |
| flatten | 16 × 128 = **2048** | 8 × 128 = **1024** | `model.py:106,147` |
| concat + dropout 0.5 | **3072**-d epoch feature | | `model.py:157-169` |
| Shortcut | FC 3072→1024 (no bias), BN, ReLU | | `model.py:281-289`; paper eq. 6 |
| Sequence | Reshape to `[batch=10, seq=25, 3072]`. Two separate 2-layer LSTM stacks, forward and backward, each with `LSTMCell(512, use_peepholes=True)` and `DropoutWrapper(output_keep_prob=0.5)` on every layer's output, run through `static_bidirectional_rnn`. The two directions are concatenated only at the end, giving 1024 | `model.py:297-347`; paper §II-B ("outputs from forward and backward LSTMs are not connected to each other") |
| Residual add | BiLSTM output + shortcut, then dropout 0.5 | `model.py:358-372`; paper eq. 6 |
| Output | FC 1024→5 (bias initialised to 0), softmax | `model.py:391` |
| Return | All 25 positions (`return_last=False`) | `train.py:64`, `model.py:342-346` |
| Init | Conv/FC: truncated normal, std = sqrt(2/fan_in) (He). BN gamma ~ N(1, 0.002), beta 0. LSTM: TF defaults (**inferred**: Glorot-uniform kernel, `forget_bias = 1.0`) | `nn.py:13-46,190-195` |
| BN | eps 1e-5, moving-average decay 0.999 (PyTorch `momentum = 0.001`). Moving statistics are used at test time | `nn.py:179-227`; paper §IV-D |
| Parameters | 22.65 M: CNNs 0.60 M, shortcut 3.15 M, BiLSTM with peepholes 18.89 M, head 5 k. The pretraining softmax (3072→5) adds 15 k and is discarded (**inferred**, computed from the shapes above) | |

### 3.3 Training (`train.py`, `deepsleep/trainer.py`, `deepsleep/optimize.py`)

| Item | Pretraining (DeepFeatureNet = CNNs + temporary softmax) | Fine-tuning (whole DeepSleepNet) | Source |
|---|---|---|---|
| Data | Training subjects' epochs, **class-balanced by oversampling** (each class repeated, then randomly topped up to the size of the largest class), shuffled every epoch, last partial batch dropped | Not oversampled. Each subject's night is split into **10 contiguous sub-sequences**, fed **25 epochs** at a time (250 epochs per step). Subjects come in fixed sorted order with no shuffling | `data_loader.py:134-137`, `utils.py:34-81`, `trainer.py:161-165`; `utils.py:105-128`, `trainer.py:485-498`; paper §III-A/B, §IV-D |
| Batch | 100 epochs | 10 × 25 | `train.py:39,59,62` |
| Optimizer | Adam lr **1e-4**, β 0.9 / 0.999, eps 1e-8 | Adam lr **1e-6** for the pretrained CNN variables and **1e-4** for the new ones (shortcut, LSTM, head) | `trainer.py:252-256`; `trainer.py:596-607`, `optimize.py:33-71` |
| Gradient clipping | none | global norm **10** | `trainer.py:252` (plain `adam`); `trainer.py:606`; paper §IV-D |
| Epochs | **100** | **200** | `train.py:25-28`; paper §IV-D |
| Loss | mean cross-entropy + L2 | **sum** of cross-entropy over 250 epochs ÷ 10 (sum over the sequence, mean over the batch) + L2 | `model.py:198-218`; `model.py:400-422` |
| L2 weight decay | λ = 1e-3 · `tf.nn.l2_loss` (= ½‖W‖²). In the code it applies **only to conv1 of CNN-1** (`wd=1e-3`); CNN-2 conv1 gets `wd=0`. The paper says "first layers of the two CNNs" (**discrepancy**) | same | `model.py:75` vs `model.py:116`, `nn.py:41-44`; paper §III-C |
| LSTM state | n/a | **Stateful**: the final forward and backward states of one step become the initial states of the next step within a night. They are reset to zero at each new subject or night (truncated BPTT over 25 epochs) | `trainer.py:491-524`; paper §II-B, Alg. 1 line 13 |
| Dropout | 0.5 after pool1 of each branch and after concat | Same, plus 0.5 on each LSTM layer's output and after the residual add | `model.py`; `trainer.py:555-558` |
| Model selection | **None.** The weights after the last epoch are used ("no early stopping as there was no validation set"). The held-out subject is evaluated each epoch for monitoring only | Same; `predict.py` restores `latest_checkpoint` | `trainer.py:417-442,784-809`, `predict.py:623-631`; paper §IV-D |
| Pretrain → finetune transfer | All pretrained variables are copied, including BN moving statistics. Adam slots and the pretraining softmax are skipped | | `trainer.py:661-679` |

### 3.4 Evaluation

| Item | Native | Source |
|---|---|---|
| Split | k-fold over subjects (Sleep-EDF k = 20, one subject with both nights per fold; MASS k = 31). No validation set | paper §IV-B; `data_loader.py:86-102`, `predict.py:690-698` |
| Inference | batch 1, seq 25, stateful across the whole night, bidirectional within each 25-epoch chunk. The model therefore sees up to 24 **future** epochs | `predict.py:604-612,510-541` |
| Epochs scored | Every epoch kept by `prepare_physionet.py`: MOVE/UNK removed, wake trimmed to ±30 min. The last `n mod 25` epochs of each night are **not scored**, because `iterate_batch_seq_minibatches` drops the remainder (**inferred** from the code) | `utils.py:105-128`, `predict.py:500-503` |
| Metrics | Predictions pooled over all folds, then ACC and MF1 (`summary.py`). κ is reported in the paper but not computed in the code | `predict.py:681-694`, `summary.py:76-78`; paper §IV-C |

## 4. Native setting on ANPHY (adaptation choices)

ANPHY facts that were checked here:

- **EDF.** Each night has 93 channels at 1000 Hz in µV, including `FPZ` and `CZ` (both
  referential), plus `EOG1/2`, `ChEMG1/2`, `RLEG±`, `LLEG±`, `ECG1/2`. Header high-pass is 0
  and low-pass 500 Hz.
- **Hypnograms.** These are `stage onset duration` rows. All 28 nights have contiguous
  30 s rows, labelled W, N1, N2, N3, R or L. `L` (lights on/off) occurs **only before the
  first or after the last sleep epoch**, never inside the sleep period.
- **Epoch counts.** There are 23,407 scored five-class epochs in total. Applying the native
  30-min wake trim would remove **287** of them, in 7 nights: EPCTL04 48, EPCTL12 72,
  EPCTL13 15, EPCTL17 21, EPCTL23 17, EPCTL26 50 and EPCTL29 64.

| Choice | ANPHY native setting | Origin |
|---|---|---|
| Channel | `FPZ − CZ` computed from the raw EDF in µV. The bipolar difference cancels the amplifier's common reference, so the recording reference does not matter | **adaptation**: Sleep-EDF stores Fpz-Cz as a hardware derivation; ANPHY has both electrodes, so the derivation is formed in software |
| Other signals | none: EEG only, single channel | paper/code |
| Software filter | none (no band-pass, no notch) | paper §IV-A / code |
| Resampling | 1000 → 100 Hz with `scipy.signal.resample_poly(x, up=1, down=10)` (default Kaiser-window FIR, β = 5, zero-phase, anti-aliasing at the new Nyquist of 50 Hz). Applied to the whole-night bipolar signal in float64 **before** epoching, then cast to float32. Rows of 30 s × 1000 samples map to exactly 3000 samples | **adaptation**, needed because the native kernels (50 / 400) and `input_dims = 3000` assume 100 Hz. `mne.filter.resample` would also do; we use `resample_poly` because it is deterministic, polyphase and chunkable. This runs only in the baseline's own loader with its own cache. The repo invariant "preprocessing never resamples" governs the GraphChangeNet pipeline, whose cache key must not change |
| Normalization | none (raw µV) | code |
| Labels | W 0, N1 1, N2 2, N3 3, R 4 (AASM already, nothing to merge) | paper/code mapping |
| `L` epochs | dropped, the same treatment as MOVE/UNK. Because they are only at the edges of the night, dropping them creates no discontinuity inside the night | code `prepare_physionet.py:126-158` |
| Wake trim | keep W only within 60 epochs of the first and last non-W epoch, for **training** subjects | code `prepare_physionet.py:187-198` |
| Test epochs | **Primary:** every scored five-class epoch of the test nights (no wake trim), to match our protocol's denominator. **Secondary:** wake-trimmed, which is the native denominator. Report both | **adaptation** |
| Artifact rejection | none. Unlike our protocol, the >500 µV training-epoch drop is **not** applied | code |
| Split | Fold 0 of our final protocol only (`baselines/folds/fold0.json`, from Exp 15 fold 0: 18 training, 5 validation and 5 test nights), for comparability. Validation nights give a logged sanity score only | **adaptation**; the paper used 20-fold CV with no validation set |
| Training | Exactly as in §3.3: oversampled pretraining (100 epochs, batch 100, Adam 1e-4), then sequential fine-tuning (200 epochs, 10 × 25, Adam 1e-6 / 1e-4, clip 10), L2 1e-3 on CNN-1 conv1 only, last-epoch weights | code. Note that ANPHY's training size (about 19k epochs) is below the paper's stated threshold of about 54k |
| Inference | Stateful bidirectional over 25-epoch chunks of the whole night, as in `predict.py`. We also score the final `n mod 25` epochs by running a shorter last chunk (PyTorch LSTMs accept any length) | code + **adaptation** for the tail |
| Seeds | Our run seeds. Upstream sets none: `predict.py` has `np.random.seed` commented out | **adaptation** |

The native setting is **non-causal**: the BiLSTM sees up to 24 future epochs. It must not be
reported next to causal models without saying so.

## 5. PyTorch implementation

### 5.1 Candidates checked

| Candidate | Commit / version | License | What it implements | Verdict |
|---|---|---|---|---|
| `braindecode.models.DeepSleepNet` | v2.0.0 (`64888d7`), identical to main `2482cb5` | BSD-3-Clause | Both CNN branches and an `nn.LSTM(bidirectional=True, num_layers=2)` with the residual FC. But **sequence length is fixed to 1** (`add_seq_dim`), so the BiLSTM runs within one epoch and never across epochs. Its docstring says so | Not faithful (details in §5.2) |
| `torcheeg.models.DeepSleepNet` | main `4fbe88b` | MIT | CNN part only (DeepFeatureNet): no LSTM, no shortcut. With more than one electrode, conv1 of CNN-1 spans the electrodes while CNN-2 does not | Not faithful |
| gist-ailab/SleePyCo `models/deepsleepnet.py` | `8ebfb13` | MIT | A modified backbone: CNN-2 uses k = 8 instead of 6, adds 1×1 compress and 3-tap smooth convs, interpolates branch 2 to branch 1. No LSTM | Not faithful |
| dongyyyyy/DeepSleepNet_pytorch | `7091443` | none | CNN only; pool2 is defined but never applied; hard-coded FC of size 9088 | Not faithful, no license |
| sujaybokil/deepsleepnet | `146f652` | none | Not valid Python (`def super(...)`, `num_features=?`); archived, README says incomplete | Unusable |
| jeonggunlee/DeepSleepNet_pytorch | `e2fbb0e` | none | notebooks only (a SleepEEGNet CNN) | Unusable |

### 5.2 braindecode in detail

- **Constructor:**
  ```python
  DeepSleepNet(n_outputs=5, return_feats=False, n_chans=None, chs_info=None, n_times=None,
               input_window_seconds=None, sfreq=None, activation_large=nn.ELU,
               activation_small=nn.ReLU, drop_prob=0.5, bilstm_hidden_size=512,
               bilstm_num_layers=2, small_first_kernel_size=50, small_first_stride=6, ...)
  ```
- **Input and output:** input `[B, n_chans, n_times]` (unsqueezed to `[B, 1, n_chans, n_times]`),
  output `[B, n_outputs]`.
- **Sampling rate:** `sfreq` is only metadata. Kernel sizes, strides and paddings are fixed in
  samples (50/6/22 and 400/50/175), so at 1 kHz they must be passed by hand.
- **Channel handling:** `Conv2d(1, 64, (1, k))`, so the **same 1-D filters are applied to
  every channel separately** and the result is flattened, giving
  `feat = n_chans × 3072`. With 83 channels that is 254,976 inputs, about 1.05 B LSTM
  parameters and about 261 M in the shortcut, which is infeasible.
- **Other differences from upstream:**
  - The large-path activation defaults to ELU (upstream uses ReLU).
  - No peepholes.
  - The PyTorch bidirectional stacking feeds layer 2 with **both** directions of layer 1,
    whereas TF runs two independent 2-layer stacks.
  - LSTM dropout is applied only between layers, not on the last LSTM output before the add.
  - BN momentum is 0.1 (upstream decay 0.999, which is momentum 0.001).
  - Kaiming init.
  - Output lengths match TF "SAME" (500 → 63 → 16 and 60 → 15 → 8), but pool2 padding is
    symmetric where TF pads only on the right.
- **Requirements:** braindecode v2.0.0 requires Python ≥ 3.11 and einops. It is not
  installed in `.venv`.

### 5.3 Chosen: our own faithful port (specification)

Because no existing port is faithful, the baseline is a dependency-free PyTorch module
(torch only, already a repo dependency). It follows `upstream/deepsleep/model.py` exactly,
apart from the deviations listed after the table. The layer list below uses
`Fs = sampling rate`, `C = input channels` and `B × T` epochs per step, and gives shapes for
100 Hz / C = 1 and 1 kHz / C = 83.

| # | Layer | Native shape (100 Hz, C = 1) | Ours (1 kHz, C = 83) |
|---|---|---|---|
| 0 | input, per epoch | `[B·T, 1, 3000]` | `[B·T, 83, 30000]` |
| 1a | CNN-1 conv1: `Conv1d(C, 64, k=Fs//2, stride=Fs//16, bias=False)`, TF-SAME pad (left = total//2), BN(eps 1e-5, momentum 0.001), ReLU | k 50, s 6 → `[·, 64, 500]` | k 500, s 62 → `[·, 64, 484]` |
| 1b | MaxPool 8/8 (SAME, −inf pad), Dropout 0.5 | 63 | 61 |
| 1c | 3 × [Conv1d(·, 128, 8, SAME, no bias), BN, ReLU] | `[·, 128, 63]` | `[·, 128, 61]` |
| 1d | MaxPool 4/4 (SAME), flatten | 16 → 2048 | 16 → 2048 |
| 2a | CNN-2 conv1: `Conv1d(C, 64, k=4·Fs, stride=Fs//2, bias=False)`, SAME, BN, ReLU | k 400, s 50 → 60 | k 4000, s 500 → 60 |
| 2b | MaxPool 4/4, Dropout 0.5 | 15 | 15 |
| 2c | 3 × [Conv1d(·, 128, 6, SAME, no bias), BN, ReLU] | `[·, 128, 15]` | same |
| 2d | MaxPool 2/2, flatten | 8 → 1024 | 8 → 1024 |
| 3 | concat, Dropout 0.5 | `[B·T, 3072]` | `[B·T, 3072]` |
| 4 | shortcut: Linear(3072, 1024, bias=False), BN1d, ReLU | `[B·T, 1024]` | same (or 512 in the causal variant, §6) |
| 5 | forward stack: LSTM(3072→512), Dropout 0.5, LSTM(512→512), Dropout 0.5. Backward stack: the same, on the time-reversed sequence. Concatenate | `[B, T, 1024]` | same |
| 6 | add (4) + (5), Dropout 0.5, Linear(1024, 5) | `[B·T, 5]` | same |
| P | pretraining head: Linear(3072, 5) on (3), discarded after pretraining | `[B·T, 5]` | same |

**Parameter counts** (computed with the shapes above, TF variant with peepholes):

| Setting | Total | CNNs | Shortcut | LSTM | Head |
|---|---|---|---|---|---|
| Native (100 Hz, 1 channel) | **22.65 M** | 0.60 M | 3.15 M | 18.89 M | 5 k |
| 1 kHz, 1 channel | 22.90 M | | | | |
| 1 kHz, 83 channels | **46.52 M** | 24.48 M | 3.15 M | 18.89 M | 5 k |

At 1 kHz with 83 channels, the two conv1 layers alone hold 2.66 M (CNN-1) and 21.25 M
(CNN-2) parameters.

**What `model.py` implements to stay faithful:**

- `nn.LSTM` has no peepholes, so `model.PeepholeLSTM` is a custom cell that follows TF
  `LSTMCell(use_peepholes=True)`. It uses TF's gate order (i, j, f, o), one bias vector,
  3 × 512 peephole weights, and a constant `forget_bias = 1.0`. With the peepholes zeroed it
  reproduces `nn.LSTM` exactly (checked), and the parameter count matches TF: **22,645,637**.
- Weight init and BN settings follow §3.2: He truncated normal, BN momentum 0.001,
  gamma ~ N(1, 0.002). The LSTM kernel and peepholes use Glorot-uniform, which is TF's
  default variable initialiser (**inferred**).

**Deviations that remain:**

- PyTorch BN updates its running variance with the unbiased batch variance, where TF uses
  the biased one.
- PyTorch Adam places epsilon slightly differently from TF1 Adam.
- Both are negligible (**inferred**).
- The training recipe of §3.3 (two optimizers or parameter groups, clipping only in
  fine-tuning, sum-over-sequence loss, L2 only on CNN-1 conv1 via an explicit
  `1e-3 · 0.5 · ‖W‖²` term rather than AdamW) is part of "their implementation" and must be
  ported with the model.

## 6. Our configuration (83 EEG at 1 kHz, causal 20-epoch runs)

**Our setting:**

- 83 scalp EEG channels at 1000 Hz, never resampled.
- Preprocessing: 0.5–45 Hz band-pass, 60 Hz notch, common average reference, RobustScaler
  fitted on training subjects and clipped at ±20.
- Items are runs of 20 consecutive scored epochs. Training runs tile each night and are
  labelled at every clean epoch. Validation and test runs end at each scored epoch and are
  labelled only there, so every test epoch is predicted from itself and the 19 epochs before
  it.
- EEG only.

**What changes in DeepSleepNet, and why:**

1. **Filter sizes scale with Fs, as the paper prescribes** (§II-C, and the commented MASS
   lines `model.py:74,115`).
   - CNN-1 conv1: k = Fs/2 = **500**, stride Fs/16 = 62.5, taken as **62**. This floors the
     same way 6.25 became 6 at 100 Hz.
   - CNN-2 conv1: k = 4·Fs = **4000**, stride Fs/2 = **500**.
   - Pools and conv2–4 stay in samples, as upstream. The flattened features are unchanged
     (2048 + 1024 = 3072) because the strides absorb the 10× rate, so everything after the
     CNNs is unchanged.
   - (**Inferred**) The 500 ms / 4 s receptive fields are preserved in seconds, but the
     signal is band-limited to 45 Hz, so conv1 is heavily oversampled. That is expected
     under the no-resampling premise.
2. **Input channels go from 1 to 83 in conv1 of both branches.** These are standard
   multi-channel `Conv1d` layers that mix electrodes in the first layer.
   - This raises the parameters from 22.65 M to **46.52 M**. Almost all of the increase is
     CNN-2 conv1 (21.25 M), so the parameter count must be reported.
   - Upstream's L2 term covers only CNN-1 conv1. Keeping that literally leaves the largest
     layer unregularised. We keep it as upstream and flag it, rather than silently adding
     decay.
   - The per-channel shared-filter alternative (braindecode style) is infeasible at 83
     channels: about 1.3 B parameters.
3. **Causality.** Native DeepSleepNet is bidirectional and stateful over the whole night, so
   it sees future epochs. Within a 20-epoch run, a BiLSTM would let epoch *t* see epochs
   *t+1 … t+19*. There are three options.
   - **(A) Unidirectional (recommended main baseline, "DeepSleepNet-causal").**
     - Keep only the forward 2-layer LSTM (512). The shortcut FC becomes 3072→512 so it can
       be added to the 512-d output. Trained at all positions of the training runs, it
       matches our protocol directly.
     - Deviations: the backward stack is removed and the residual width is 512 instead of
       1024. The LSTM has 9.44 M parameters instead of 18.89 M, and the shortcut 1.57 M
       instead of 3.15 M, for about 35.5 M in total at 83 channels (**inferred** from the
       shapes).
     - History is truncated to 19 epochs, versus unbounded stateful history in the native
       model.
   - **(B) Bidirectional, last epoch only.**
     - Keep the exact BiLSTM, but put the loss **and** the prediction only at the last
       position of each 20-epoch run. There the backward direction has seen only epoch *t*,
       so nothing leaks.
     - Training items must then also be "scored epoch + 19 predecessors" with the loss at
       the end. That gives one label per run instead of 20, so stage 2 costs about 20× more
       forward passes per label.
     - Training at all positions but scoring only the last one has a train/test mismatch:
       the backward path sees 0–19 future epochs during training and 0 at test. The run
       `ours` (§8) is set up exactly this way, by decision, because it uses our trainer and
       loaders unchanged. That mismatch is its main caveat.
     - B keeps the architecture literally faithful, but the backward half of the BiLSTM
       contributes a single step.
   - **(C) Native non-causal.** A BiLSTM over runs labelled at every position. This violates
     the causal claim. Use it only as a clearly labelled non-causal upper reference, if at
     all.
4. **Training recipe.** The native recipe (§3.3) would be kept with these changes. The run
   `ours` (§8) instead uses our stage-2 trainer from scratch, by decision.
   - Pretraining is per epoch (`T = 1`) with oversampling.
   - Fine-tuning runs on our 20-epoch runs instead of stateful 25-epoch chunks. Each run
     starts from a zero state; that is the adaptation.
   - Our >500 µV training-epoch drop and our subject split apply. Native has no early
     stopping, so either train for fixed epochs and take the last weights (faithful), or use
     our validation-based selection (comparable). Say which.
5. **Normalization and filtering.** Our band-pass, notch, CAR and RobustScaler replace
   native "raw µV, no filter". Inputs are then roughly unit-scale instead of tens of µV.
   BN after every conv absorbs this (**inferred**).
6. **Memory.** Measured on an A100 80 GB with fp32, forward + backward + Adam step, using a
   scratch PyTorch model of §5.3 (`nn.LSTM`, not upstream code). The measured numbers for
   the real port are in §8:

   | Setting | Peak memory |
   |---|---|
   | Native (100 Hz, 1 channel), batch 10 × 25 | 0.76 GiB |
   | Ours (1 kHz, 83 channels), batch 1 run × 20 | **1.63 GiB** |
   | Ours, batch 4 × 20 | 4.05 GiB |
   | Ours, batch 10 × 20 | 9.16 GiB |

   One run is 20 × 83 × 30000 fp32 = 199 MB of input (100 MB in the fp16 cache), and the
   input dominates the activations. The native batch of 10 sequences therefore fits.
   Pretraining with batch 100 single epochs is about 1 GB of input (**inferred**).
   Throughput and data I/O, not memory, are the constraints.

## 7. Files

| Path | Content |
|---|---|
| `upstream/` | Official TF code, `git clone` of akaraspt/deepsleepnet at `eb72f9b`. Read only, never executed |
| `paper.pdf` | arXiv 1703.04046v2 |
| `README.md` | this file |
| `model.py` | PyTorch port, with the sampling rate as a parameter (§5.3) |
| `runner.py` | `run(spec, settings, out_dir, quick)` for `native` and `ours` |
| `native.yaml`, `ours.yaml` | the two configurations |
| `__init__.py` | package marker |

## 8. How to run and implementation decisions

Run from the repo root:

```bash
.venv/bin/python -m baselines.run deepsleepnet native [--quick]
.venv/bin/python -m baselines.run deepsleepnet ours [--quick]
```

- **Outputs** go to `baselines/results/deepsleepnet/<config>[_quick]/`: `config.yaml`,
  `predictions.npz` (one prediction per scored test epoch) and `metrics.json`, plus the
  weights.
- **Native caches** go to `baselines/cache/deepsleepnet/native[_quick]/`, one resampled
  FPZ−CZ night per subject.
- **Split:** both configurations use fold 0 only (`baselines/folds/fold0.json`): 18
  training, 5 validation and 5 test nights, taken from Exp 15 fold 0.

**`native`: their configuration (coordinator's decisions).**

- **Signal.** FPZ − CZ from the EDF in µV, read through `baselines.common.native.read_channels`.
  It is resampled once per whole night with `scipy.signal.resample_poly(x, 1, 10)` to
  100 Hz. No filter, no normalization.
- **Epochs.** Epochs are cut at our pipeline's hypnogram onsets
  (`baselines.common.native.night_epochs`). Rows without a stage (`L`, lights on) are
  dropped.
- **Training nights.** Wake is trimmed to 60 epochs (30 min) around sleep, only in the
  training nights.
- **Validation and test nights.** Every scored epoch is kept.
- **Training.** Exactly upstream's two steps:
  - Pretraining of the CNNs with their own softmax: class-balanced oversampling (indices
    drawn once), shuffled batches of 100 with the last partial batch dropped, Adam 1e-4,
    mean cross-entropy, 100 epochs, no clipping.
  - Fine-tuning: 200 epochs. Each training night, in fold order, is split into 10 contiguous
    sub-sequences read 25 epochs at a time (the remainder is dropped, as upstream). The state
    of both directions is carried over and detached between steps, and reset at each night.
    Adam runs at 1e-6 on the CNN and 1e-4 on the rest, with global-norm clipping at 10, and
    the loss is summed cross-entropy ÷ 10.
- **Weight decay.** As in the code, not the paper: 1e-3 · ½‖W‖² on conv1 of CNN-1 only.
- **Model selection.** None. The last-epoch weights are saved as `deepsleepnet_last.pt` and
  used. Validation nights only give a logged sanity score (`val_sanity` in `metrics.json`).
- **Test.** Every scored epoch of the 5 test nights. Each night is read as one sequence,
  stateful and bidirectional, 25 epochs at a time, and the final partial chunk is included
  (upstream drops it).
- **Scoring.** By `baselines.common.scoring.score_run`.

**`ours`: our configuration (coordinator's decisions).**

- **Training and data.** Our data and our stage-2 trainer, through
  `baselines.common.ours.run_ours`:
  - Input is 83 EEG channels at 1 kHz: notch, 0.5–45 Hz, CAR, robust scaling clipped at ±20.
  - Training uses runs of 20 epochs labelled at every clean position.
  - There is one test item per scored epoch, ending at it.
  - The model trains from scratch with the override `finetune.freeze_epochs: 0` (as
    Exp 15a).
- **Architecture.** Upstream's, with only the forced input changes:
  - Kernels and strides scale with Fs = 1000: CNN-1 conv1 k500 / s62, CNN-2 conv1 k4000 / s500.
  - Both conv1 layers take 83 input channels.
  - The **BiLSTM is kept**.
- **Causality.** The labelled test epoch is the last position of its run, so no later epoch
  reaches it.
- **Caveat.** Training puts the loss on all 20 positions. During training the backward
  direction sees up to 19 later epochs, while at test it sees none (§6, option B).
- **Not applied.** Upstream's L2 term, oversampled pretraining, two learning rates and
  clipping at 10 belong to their recipe, not to our trainer, and are not applied here.
- **Parameters:** **46,520,837** (native: **22,645,637**; the pretraining head adds 15,365
  and is discarded).

**Port checks.** These run without data.

- **Layer output lengths** match §5.3:
  - 100 Hz: 500 → 63 → 63 → 16 (2048) and 60 → 15 → 15 → 8 (1024).
  - 1 kHz, 83 channels: 484 → 61 → 61 → 16 and 60 → 15 → 15 → 8, so the feature size is
    3072 in both.
- **Parameter count** of the native network is 22,645,637, the same as the TF graph.
- **TF "SAME" padding** gives (22, 22) for conv1 of CNN-1 at 100 Hz, and (0, 1) for its
  last pooling.
- **Peephole LSTM:** with the peepholes zeroed it equals `nn.LSTM`.
- **Stateful step:** a run split into two stateful steps gives the same forward-direction
  logits for its first chunk.

**Smoke tests** (`--quick`, fold 0, A100 shared with other agents). Both finished and wrote
`metrics.json`; their outputs and caches were deleted afterwards. The scores are
meaningless at this size.

- **`native --quick`:** 3 training nights (2,730 epochs after the wake trim; 6,340
  oversampled), 1 validation and 1 test night (886 test epochs, all scored). One epoch of
  each step, 4 steps each. Wall time 3 min 3 s, of which 172 s went to reading the EDFs and
  resampling five nights (about 35 s per night, cached afterwards).
- **`ours --quick`:** 2 training, 1 validation and 1 test subject, 48 validation and test
  items, 1 epoch. Wall time 2 min 5 s, with 56 s for the training epoch. Peak GPU memory
  2.58 GiB; the trainer uses batch 2 runs × 20 epochs and an EMA copy.

**Steady-state timings and peak memory** (A100, fp32, synthetic input, the real port):

| Step | Time | Peak memory |
|---|---|---|
| native pretraining step (batch 100 epochs) | 6.8 ms | 0.12 GiB |
| native fine-tuning step (10 × 25, peephole BiLSTM) | 92 ms | 0.61 GiB |
| native inference (1 × 25 chunk) | 24 ms | 0.37 GiB |
| ours training step (2 runs × 20, AdamW) | 94 ms | 2.41 GiB |
| ours training step (4 runs × 20) | 111 ms | 4.02 GiB |
| ours inference (2 runs × 20) | 20 ms | 2.00 GiB |
| ours inference (4 runs × 20) | 29 ms | 3.21 GiB |

**Full-run estimates** (**inferred** from the step times):

- **Native:**
  - Pretraining: about 300 steps per epoch × 100 epochs, a few minutes.
  - Fine-tuning: about 3 steps per night × 18 nights per epoch × 200 epochs, about 20 min.
  - Loading 28 nights once: about 16 min.
- **Ours:** the quick run's training epoch took about 1.3 s per step against 0.09 s of GPU
  compute, so it is bound by reading the 1 kHz cache, as our own model is. Full epochs are
  dominated by data loading.
