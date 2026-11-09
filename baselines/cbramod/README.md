# CBraMod baseline (setup notes)

This folder holds the reference material and the runner for the CBraMod baseline on
ANPHY-Sleep. The upstream code was read but never executed. The runner uses our own port of
the three upstream model files (see "How to run / implementation decisions" at the end).

Line references point at `upstream/` at commit `b9e9610` unless another file is named.
Anything marked **(inferred)** is our reading or our estimate. Neither the paper nor the code
states it.

## Citation and links

Jiquan Wang, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, Gang Pan.
*CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.* ICLR 2025.
arXiv:2412.07236 · OpenReview `NPNUHgHF2w`.

```bibtex
@inproceedings{wang2025cbramod,
  title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
  author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

- Code: https://github.com/wjq-learning/CBraMod (MIT license, `upstream/LICENSE`)
- Weights: https://huggingface.co/weighting666/CBraMod (model card license: apache-2.0)
- Paper: `paper.pdf` (arXiv version, 2,255,209 bytes)

## Paper summary

**Problem.** Earlier EEG foundation models (BIOT, LaBraM) flatten all channel×time patches
into one token sequence and use full attention over it. The authors argue that spatial
(across-channel) and temporal (within-channel) dependencies are different in kind, and that
full attention is costly when there are many patches. Downstream datasets also differ in
channel set and duration, which fixed positional embeddings handle poorly.

**Method.**
- *Patching.* An EEG sample `C × T` at 200 Hz is cut into 1 s patches of 200 points, giving
  `X ∈ R^{C×n×200}`.
- *Time–frequency patch encoder.* There are two branches. The time branch is a 3-layer
  conv (kernel 49/3/3, stride 25/1/1, 25 filters, GroupNorm, GELU), giving 25×8 = 200
  features. The frequency branch takes `|rfft(patch, norm='forward')|`, which is 101 bins at
  1 Hz from 0 to 100 Hz, through `Linear(101, 200)`. The two are summed.
- *ACPE (asymmetric conditional positional encoding).* A depthwise `Conv2d(200, 200,
  kernel=(19, 7), padding=(9, 3))` over the (channel, patch) grid, added to the
  embeddings. It is not tied to a fixed number of channels or patches.
- *Criss-cross transformer.* 12 pre-norm layers with d = 200 and FFN 800. Each layer splits
  the 200 features in half. The first 100 go through S-attention: 4 heads over the
  channels at each patch position. The other 100 go through T-attention: 4 heads over the
  patches within each channel. The two outputs are concatenated, so S and T run in
  parallel inside one layer.
- *Pretraining objective.* Each patch is masked independently with probability 0.5. The
  mask token is all zeros. The training target is the MSE on the masked patches only,
  through a `Linear(200, 200)` reconstruction head.

**Pretraining data.** TUEG: 69,652 recordings, 14,987 subjects, 27,062 h. Preprocessing
keeps recordings longer than 5 min, drops the first and last minute, and keeps 19 10-20
channels (code order: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ,
CZ, PZ). It then resamples to 200 Hz, filters 0.3–75 Hz, applies a 60 Hz notch, and cuts
the data into 30 s non-overlapping samples. Any sample with |x| > 100 µV is dropped. Units
are 100 µV, i.e. x/100. That leaves 1,109,545 samples (more than 9,000 h). Training ran 40
epochs at batch 128 with AdamW, lr 5e-4, weight decay 5e-2 and cosine schedule, on 4×A5000
for about 5 days.
Sources: paper §3.1 and Table 5; `preprocessing/preprocessing_tueg_for_pretraining.py:11-27,
65-89`; `pretrain_main.py:22-43`; `pretrain_trainer.py:17, 35-38, 64, 66-73`.

**Main results** (mean ± std over 5 seeds):

| Dataset | Metric | CBraMod | best other baseline |
|---|---|---|---|
| **ISRUC (sleep, 5-class), Table 7** | BAcc | **0.7865 ± 0.0110** | LaBraM-Base 0.7633 |
| | Cohen's κ | **0.7442 ± 0.0152** | LaBraM-Base 0.7231 |
| | weighted F1 | **0.8011 ± 0.0099** | USleep 0.7805 / LaBraM 0.7810 |
| FACED (9-class), Table 2 | κ | 0.5041 ± 0.0122 | LaBraM 0.4698 |
| SEED-V (5-class), Table 2 | κ | 0.2569 ± 0.0143 | LaBraM 0.2386 |

ISRUC is the only sleep dataset in the paper. Other Table 7 rows for sleep: USleep has BAcc
0.7586, κ 0.7209, wF1 0.7805. DeepSleepNet has κ 0.7036. BIOT has κ 0.7192. All of them were
re-run with the same 20-epoch sequence encoder. The paper lists CBraMod at "4.0M" parameters
(Tables 7 and 22, counted with Thop). Our count is different; see "Parameter count" below.

## Native configuration (ISRUC sleep staging)

| Item | Native setting | Source |
|---|---|---|
| Dataset | ISRUC-Sleep subgroup 1: 100 subjects, one night each, 89,240 epochs of 30 s | paper App. E.1; `preprocessing/ISRUC/prepare_ISRUC_1.py:13,20-23` |
| Labels | scorer 1 (`{n}_1.txt`). Map `0→W, 1→N1, 2→N2, 3→N3, 5→REM(4)`. Any other code raises a KeyError | `prepare_ISRUC_1.py:23,35-39,84-87` |
| Channels | 6 EEG: F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1 (contralateral mastoid reference). **No EOG, EMG or ECG.** | paper App. E.1; selected by column `psg_array[:, 1:][:, 2:8]`, `prepare_ISRUC_1.py:60-61`. That this column slice is exactly these six depends on the ISRUC `.rec` channel order (LOC-A2, ROC-A1, F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1, X1, …) **(inferred)** from the ISRUC documentation and the commented `signal_name = ['LOC-A2','F4-A1']`, line 42 |
| Sampling rate | 200 Hz native. **No resampling** (`raw.resample(sfreq=200)` is commented out) | `prepare_ISRUC_1.py:52`; paper Table 1 |
| Band-pass | 0.3–35 Hz, MNE `raw.filter(0.3, 35, fir_design='firwin')`. This is MNE's default zero-phase, non-causal FIR | `prepare_ISRUC_1.py:53` (output dir `precessed_filter_35`, line 15) |
| Notch | 50 Hz, `raw.notch_filter(50)`. It is applied after the 35 Hz low-pass, so it does almost nothing | `prepare_ISRUC_1.py:54` |
| Units / scaling | `raw.to_data_frame()` gives µV (MNE default EEG scaling **(inferred)**). The dataset then divides by 100, so the unit is 100 µV. No per-subject z-scoring (the `StandardScaler` is commented out). | `prepare_ISRUC_1.py:57,64-66`; `datasets/isruc_dataset.py:28`. **Note:** `/100` was added in commit `b4b489f` (2025-07-09), after the paper. Before that, ISRUC fed raw µV. We cannot tell which version produced Table 7. |
| Epoching | The recording is cut from t = 0 into 30 s × 200 = 6000-sample epochs. The trailing partial epoch is dropped | `prepare_ISRUC_1.py:68-72` |
| Sequence | **20 consecutive epochs, non-overlapping.** The last `n mod 20` epochs of each night are **dropped** from train, val **and test** | `prepare_ISRUC_1.py:75-80,89-91` |
| Stored sample | `seq` `[20, 6, 6000]` float, `label` `[20]` int | `prepare_ISRUC_1.py:79-80,96-107` |
| Split | by subject in index order: 1–80 train, 81–90 val, 91–100 test | `datasets/isruc_dataset.py:94-107`; paper App. E.1 |
| Dataset class | `datasets/isruc_dataset.py: LoadDataset(params)`. Train: `batch_size=64, shuffle=True`. Val and test: `batch_size=1`. `collate` returns `float x [B,20,6,6000]` and `long y [B,20]`. Seq and label files are paired with `zip(os.listdir(a), os.listdir(b))` **without sorting**, so the pairing depends on filesystem order (fragile, **inferred** to work on their FS). | `isruc_dataset.py:10-33,43-70,85-89` |
| Backbone | `CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)`. Patch 200 samples = 1 s, so 30 patches per epoch. `proj_out` is replaced by `Identity` | `models/model_for_isruc.py:11-19`; `models/cbramod.py:9-33` |
| Epoch encoder → token | backbone output `[B·20, 6, 30, 200]` → flatten `6·30·200 = 36000` → `Linear(36000, 512)` + GELU | `model_for_isruc.py:21-24,37-40` |
| Sequence model | `nn.TransformerEncoder` with 1 layer: d 512, 4 heads, FFN 2048, GELU, pre-norm, dropout 0.1 (default). **Bidirectional, no attention mask, no positional encoding over the 20 epochs.** Without positional encoding the layer is permutation-equivariant across epochs **(inferred from the code)** | `model_for_isruc.py:26-29,41` |
| Classifier | `Linear(512, 5)` on all 20 positions, output `[B, 20, 5]` | `model_for_isruc.py:30,42` |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` on `pred.transpose(1,2)` vs `y [B,20]`, so all 20 epochs are trained | `finetune_trainer.py:22-23,83-84`; `finetune_main.py:46` |
| Optimizer | AdamW, weight decay 5e-2, betas (0.9, 0.999) | `finetune_main.py:22-23`; `finetune_trainer.py:44-49`; paper Table 6 |
| LR (multi_lr=True) | backbone **1e-4**. Head, sequence encoder and classifier: **0.001·sqrt(64/256) = 5e-4** | `finetune_trainer.py:31-49`; `finetune_main.py:21,47` |
| Scheduler | `CosineAnnealingLR(T_max = 50 · len(train_loader), eta_min=1e-6)`, stepped every iteration. No warm-up | `finetune_trainer.py:63-66,94` |
| Epochs / batch | 50 epochs, batch 64 sequences (= 1,280 epochs of 30 s per step) | `finetune_main.py:19-20`; Table 6 |
| Grad clipping | `clip_grad_norm_(…, 1.0)` | `finetune_main.py:24`; `finetune_trainer.py:90-91` |
| Dropout | 0.1 in every backbone layer (attention + FFN), in `spectral_proj`, and in the sequence encoder. The `--dropout` CLI arg is **not used** by the ISRUC model | `criss_cross_transformer.py:38`; `cbramod.py:59-63`; `model_for_isruc.py` |
| Frozen? | No, full fine-tuning (`frozen=False`) | `finetune_main.py:49-50`; `finetune_trainer.py:37-40` |
| Pretrained init | `pretrained_weights/pretrained_weights.pth` (HF), loaded strictly into the backbone before `proj_out` is replaced | `model_for_isruc.py:16-19`; `finetune_main.py:51-55` |
| Seed | default 3407. The paper reports 5 seeds without naming them | `finetune_main.py:17,144-149`; paper §3.2 |
| Model selection | best **validation Cohen's κ** over 50 epochs, then a single test pass with that state | `finetune_trainer.py:98-128`; paper §3.2 |
| Metrics | `balanced_accuracy_score`, `cohen_kappa_score`, `f1_score(average='weighted')`, confusion matrix. Pooled over all test epochs (all subjects concatenated) | `finetune_evaluator.py:13-34` |
| ISRUC run command | **not in the repo.** `finetune_main.py` defaults to MentalArithmetic. ISRUC needs `--downstream_dataset ISRUC --num_of_classes 5 --datasets_dir <...>/precessed_filter_35`. Everything else uses the defaults above **(inferred)** | `finetune_main.py:35-42` |

argparse gotcha: `--multi_lr`, `--frozen` and `--use_pretrained_weights` are `type=bool`, so
`--frozen False` evaluates to **True** (`finetune_main.py:47-52`). Leave them at their defaults.

### Parameter count (computed from the checkpoint and the module shapes)

| Part | Parameters |
|---|---|
| Pretrained checkpoint, all tensors | 4,924,000 (patch embedding 52,600 incl. 200-dim fixed `mask_encoding`; encoder 4,831,200; `proj_out` 40,200) |
| Backbone as used (proj_out → Identity) | 4,883,800 |
| ISRUC head: `Linear(36000,512)` | 18,432,512 |
| ISRUC sequence encoder (1 layer, d 512) | 3,152,384 |
| Classifier | 2,565 |
| **Full ISRUC model** | **26,471,261 (≈ 26.5 M)** |

The paper's "4.0M" (Tables 7 and 22) is below even the backbone. **(inferred)** Thop does not
count `nn.MultiheadAttention`, and the ISRUC head is not included. Report 26.5 M for the
native ISRUC model.

## Model API

Upstream uses absolute imports such as `from models.criss_cross_transformer import ...`, so
`baselines/cbramod/upstream` has to be on `sys.path`. The folder names `models/`,
`datasets/` and `utils/` are generic. Put upstream first on the path, or import by file
path, so they do not shadow other packages. (The HF `datasets` package is not installed in
`.venv` today.)

```python
import sys; sys.path.insert(0, "baselines/cbramod/upstream")
from models.cbramod import CBraMod              # backbone
from models.model_for_isruc import Model        # ISRUC model (backbone + head + seq transformer)

backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                   seq_len=30, n_layer=12, nhead=8)   # these are also the defaults
state = torch.load("baselines/cbramod/weights/pretrained_weights.pth",
                   map_location="cpu", weights_only=True)   # verified: loads under torch 2.13 with weights_only
backbone.load_state_dict(state)                 # strict; keys match module names
backbone.proj_out = torch.nn.Identity()

# ISRUC Model takes an argparse-like object:
#   param.use_pretrained_weights (bool), param.cuda (int, map_location=f'cuda:{cuda}'),
#   param.foundation_dir (path), param.num_of_classes (5)
```

| Module | Input | Output |
|---|---|---|
| `CBraMod.forward(x, mask=None)` | `x [B, C, n_patches, 200]` (C and n are free; patch size must be 200 because the time branch reshapes to 25×8 = d_model and the spectral branch expects 101 bins, `cbramod.py:79-86`). `mask [B, C, n]` with 1 = masked (pretraining only) | `[B, C, n, 200]` (after `proj_out`, which is Identity when fine-tuning) |
| `Model.forward(x)` (ISRUC) | `x [B, 20, 6, 6000]`, values in units of 100 µV. Internally reshaped to `[B·20, 6, 30, 200]` (hard-coded 30 and 200, `model_for_isruc.py:37`) | logits `[B, 20, 5]` |

So the layout is `[B, seq, C, 6000]` at the ISRUC model. It becomes `[B·seq, C, 30, 200]`
at the backbone, not `[B, seq, C, patches, 200]`.

Requirements. `models/cbramod.py`, `models/criss_cross_transformer.py` and
`models/model_for_isruc.py` import only `torch`. `upstream/requirements.txt` adds einops,
h5py, lmdb, matplotlib, mne, numpy, pandas, ptflops, pyEDFlib, scikit_learn, scipy, torchinfo,
tqdm and umap_learn, which are needed only for their preprocessing, pretraining and logging.
The paper's environment was Python 3.11.7 with PyTorch 2.1.2 and CUDA 12.1. Our `.venv` has
torch 2.13 + cu129, mne, scipy and sklearn, so a port needs no new dependency.

## Native setting on ANPHY (our adaptation of the ISRUC setup)

ANPHY's EDF has 83 scalp EEG channels, EOG1/EOG2, ChEMG1/2, RLEG±/LLEG± and ECG1/2 at
1000 Hz, in µV (header checked on `EPCTL01 - fixed.edf`). There are **no A1/A2/M1/M2
mastoid or earlobe channels.** The recording reference is not given in the header. Bipolar
differences between two scalp channels cancel the common recording reference whatever it
was. Candidates from `Co-registered average positions.pos` (x anterior, y left, z up, cm):

| Electrode | x | y | z | note |
|---|---|---|---|---|
| TP9 | -4.37 | 7.38 | 1.90 | standard 10-10 mastoid surrogate (left) |
| TP10 | -3.17 | -7.87 | 0.90 | standard 10-10 mastoid surrogate (right) |
| TP11 | -3.64 | 7.03 | -0.76 | lower, closer to the mastoid process **(inferred)** |
| TP12 | -4.47 | -6.82 | -0.71 | lower, closer to the mastoid process **(inferred)** |
| T9 / T10 | 1.10 / 1.94 | ±8.2 | ~1.6 | pre-auricular, nearer the earlobe A1/A2 of 10-20 |

**Proposed mapping (our adaptation):** the contralateral mastoid surrogate TP9/TP10, in the
ISRUC column order.

| ISRUC | ANPHY derivation |
|---|---|
| F3-A2 | F3 − TP10 |
| C3-A2 | C3 − TP10 |
| O1-A2 | O1 − TP10 |
| F4-A1 | F4 − TP9 |
| C4-A1 | C4 − TP9 |
| O2-A1 | O2 − TP9 |

As a sensitivity variant, TP11/TP12 can replace TP9/TP10. EEG only, as in the native setup
(no EOG or EMG).

Other choices (all ours, because ISRUC needed no resampling):
- **Resampling** 1000 → 200 Hz with an exact integer factor of 5:
  `scipy.signal.resample_poly(x, 1, 5)`, which has a built-in anti-alias FIR, or MNE
  `raw.resample(200)`. Resample first, then filter, as ISRUC filtered a 200 Hz signal.
- **Filter** `raw.filter(0.3, 35, fir_design='firwin')` (native). For the **notch**, use
  60 Hz for North American mains (our project uses `notch: 60.0`, `configs/default.yaml:16`)
  instead of ISRUC's 50 Hz. It has almost no effect after the 35 Hz low-pass.
- **Units** µV, then `/100`. No subject-wise or train-fitted scaler (native).
- **Epochs** come from our scored-epoch table (`baselines/common/reference.py:scored_epochs`)
  so the epoch set is the same as the reference model's.
- **Sequences** of 20 consecutive epochs, non-overlapping, bidirectional, trained on all 20
  positions (native). Native drops the last `n mod 20` epochs of every night. Our shared
  scorer requires **every scored test epoch** to have a prediction
  (`baselines/common/scoring.py:align` raises otherwise). So the adaptation is to add a final
  right-aligned window per night, and to take each already-covered epoch's prediction from
  its first window. Breaks in the scored-epoch sequence (unscored or removed epochs) end a
  run **(inferred handling)**.
- **Split** `baselines/folds/fold0.json` (18 train / 5 val / 5 test nights, copied from
  `runs/exp15_final_fold0`) replaces ISRUC's 80/10/10.
- **Hyperparameters** native: 50 epochs, batch 64, AdamW with lr 1e-4 for the backbone and
  5e-4 for the head, weight decay 5e-2, per-step cosine down to 1e-6, clip 1, label
  smoothing 0.1, select on validation κ. ANPHY has about 23k scored epochs over 28 nights
  (from `configs/final_protocol.yaml`), so 18 training nights give about 15k epochs, about
  750 sequences, and about 12 steps per epoch. That is about 600 steps in total, against
  roughly 2,800 on ISRUC **(inferred)**.
- **Model** is the native `Model` unchanged (6 ch, so `Linear(36000, 512)`), 26.5 M params.
  Memory at batch 64 × 20 × 6 ch is about 230k backbone tokens. Our rough estimate is
  around 45 GB of fp32 activations **(inferred, measure it)**. That fits the local A100
  80 GB. Use a smaller batch with gradient accumulation if it does not.
- **Report** κ and macro-F1 first (our convention), and also the native BAcc and weighted F1.

## "Our configuration" (assessment only, not implemented)

Our configuration: 83 EEG at 1000 Hz with no resampling, 0.5–45 Hz, 60 Hz notch, common
average reference, RobustScaler fitted on the training subjects and clipped to ±20, runs of
20 consecutive 30 s epochs, causal (epoch t predicted from t-19…t), EEG only.

**1. Patch length at 1 kHz.** `PatchEmbedding` only works with 200-sample patches.
- **(a) 1 s = 1000-sample patches → 30 patches/epoch.** The tokens keep the pretraining
  meaning (1 s, 30 per epoch), but the patch encoder must change. The time branch at 1000
  samples gives `25 × 40 = 1000` features, so the `.view(..., d_model=200)` fails. The
  spectral branch gives 501 bins, so `view(..., 101)` and `Linear(101, 200)` fail. Two
  ways to keep the pretrained patch encoder **(inferred)**:
  - *Spectral branch.* With `norm='forward'`, the first 101 bins of a 1000-point rfft at
    1 kHz are exactly 0–100 Hz at 1 Hz spacing, on the same amplitude scale (A/2 for a
    sinusoid of amplitude A) as the 200-point rfft at 200 Hz. Truncating to bins 0–100
    therefore reuses `spectral_proj` unchanged. With a 45 Hz low-pass, bins above 100 Hz
    are near zero anyway.
  - *Time branch.* Needs a new first conv (kernel 245, stride 125 gives 8 positions × 25 =
    200). It can be initialised from the pretrained 49-tap kernel upsampled ×5 and scaled
    by 1/5. That is an approximation.

  With (a), the transformer, ACPE and all pretrained transformer weights apply unchanged.
  This is the most faithful no-resampling option. It is a new component, not a change to
  upstream.
- **(b) 200-sample patches = 0.2 s → 150 patches/epoch.** No code change in the backbone,
  and the weights load, but their meaning shifts. Spectral bin k now means 5k Hz (0–500 Hz
  at 5 Hz resolution), so 0.5–45 Hz collapses into bins 0–9. The conv receptive field
  shrinks from 245 ms to 49 ms. The ACPE temporal kernel of 7 spans 1.4 s instead of 7 s.
  T-attention runs over 150 tokens instead of 30. Tokens per epoch grow 5×, to 83 × 150 =
  12,450. The pretrained weights are effectively out of distribution.
  `model_for_isruc.py:37` (hard-coded `30, 200`) and the flatten size have to change.
- **(c)** A learned or fixed strided front-end from 1000 to 200 samples per second, ahead of
  the unchanged backbone, is resampling inside the model. It conflicts with the 1 kHz
  premise and is listed only for completeness.

**2. Head size at 83 channels.** The native flatten head grows with C × n × 200:

| Setting | flatten dim | `Linear(·, 512)` params | full model |
|---|---|---|---|
| native 6 ch, 30 patches | 36,000 | 18.4 M | 26.5 M |
| 83 ch, 30 patches (200 Hz, or option a) | 498,000 | 255.0 M | 263.0 M |
| 83 ch, 150 patches (option b) | 2,490,000 | 1,274.9 M | 1,282.9 M |

At 83 channels the native head is 50–260× larger than the backbone. Weights, gradients and
two AdamW moments come to about 4.2 GB (a) or 20.5 GB (b) in fp32. This would also make the
"bigger model" objection unanswerable. A faithful-in-spirit replacement **(inferred, our
choice)** pools before the `Linear`: average over channels, giving `[30·200 = 6000] →
Linear(6000, 512)`, about 3.1 M parameters, close to the native per-epoch head scale.
Alternatively, pool over both channels and patches, giving `200 → 512`.

**3. Channel ordering and ACPE.** ACPE's 19-wide depthwise kernel slides over the channel
*index*, not over scalp geometry. With 83 channels, the order decides which electrodes count
as neighbours. Pretraining used 19 channels, so one kernel covered the whole montage. Order
the 83 channels so that nearby indices are nearby on the scalp (for example the montage
order or a space-filling ordering) **(inferred)**. The kernel and padding need no change
for C = 83.

**4. Reference and scaling.** Pretraining used referential TUEG montages in units of
100 µV, with samples above 100 µV removed, so values were mostly within ±1. Our CAR plus
RobustScaler (per-electrode, IQR units) clipped at ±20 gives values a few times larger.
This is a scale shift that fine-tuning must absorb **(inferred)**. The common average
reference is compatible with the backbone because it does not assume a reference.

**5. Causality.** The backbone runs on each epoch on its own (`[B·20, C, n, 200]`), so it
does not leak information across epochs. Within an epoch, T-attention is bidirectional over
the 30 s. That is allowed, because epoch t may use all of itself. The leak is the
sequence encoder: it is bidirectional and has no mask (`model_for_isruc.py:41`). The fix is
to pass a causal mask (`torch.triu(full((20, 20), -inf), 1)`, i.e.
`nn.Transformer.generate_square_subsequent_mask(20)`) as
`self.sequence_encoder(epoch_features, mask=...)`. That also gives the encoder its only
source of order information, since there is no positional encoding.
- *Train* on all 20 causal positions. Position k then sees k+1 epochs. This matches native
  training density.
- *Score* the last position of a stride-1 sliding window, so every test epoch has its full
  19-epoch history. The first 19 epochs of a night get shorter histories, padded or
  truncated. Because the backbone is per-epoch, its features can be computed once per
  epoch and cached. Stride-1 evaluation then costs one backbone pass per epoch plus 20
  cheap passes of the 1-layer, d = 512 encoder **(inferred)**.

**6. Memory, batch 1 × 20 epochs × 83 ch** (fp32 training, rough estimate **(inferred)**).
The estimate assumes about 4.3k saved floats per token per layer, with SDPA attention so
that no attention matrices are stored. Measure with `torch.cuda.max_memory_allocated`
before planning runs.

| Setting | backbone tokens / sequence | activations (12 layers) | weights + grads + AdamW |
|---|---|---|---|
| native 6 ch @200 Hz | 3,600 | ~0.7 GB (×64 batch ≈ 45 GB) | 0.4 GB |
| 83 ch, 30 patches (option a), native flatten head | 49,800 | ~10 GB | ~4.2 GB |
| 83 ch, 30 patches (option a), pooled head | 49,800 | ~10 GB | ~0.1 GB |
| 83 ch, 150 patches (option b), native flatten head | 249,000 | ~51 GB | ~20.5 GB |

Option (a) with a pooled head fits batch 1–4 on the local A100 80 GB. Option (b) needs
activation checkpointing or bf16 even at batch 1. EEG only. The model has no slot for
EOG, EMG or ECG.

## Files

| Path | What |
|---|---|
| `upstream/` | `git clone https://github.com/wjq-learning/CBraMod`, commit **`b9e961003214326972c567eff390e75b0287e32a`** (2026-08-07, "update"). Backbone files last changed at `0e7c20b` (2025-02-20). `model_for_isruc.py` last changed at `2ac8068` (2025-06-13). `isruc_dataset.py` last changed at `b4b489f` (2025-07-09, added `/100`). MIT license. Not executed. |
| `paper.pdf` | arXiv 2412.07236 (ICLR 2025 version), 2,255,209 bytes |
| `weights/pretrained_weights.pth` | from `https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth` (HF repo revision `500543c`, last modified 2024-12-11), **19,775,842 bytes**, sha256 **`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`** (equal to the HF LFS oid). A plain state dict with 211 tensors and 4,924,000 parameters. Loads with `weights_only=True`. Upstream expects it at `upstream/pretrained_weights/pretrained_weights.pth` (`finetune_main.py:53-55`). |

## How to run / implementation decisions

```bash
.venv/bin/python -m baselines.run cbramod native     # CBraMod's ISRUC recipe on fold 0
.venv/bin/python -m baselines.run cbramod ours       # CBraMod's architecture on our configuration
# add --quick for a smoke test (settings.yaml: quick)
```

Outputs go to `baselines/results/cbramod/<config>[_quick]/`: `config.yaml`,
`predictions.npz` and `metrics.json`. `metrics.json` holds the shared scores
(`baselines/common/scoring.py`), the parameter count, the history and `resources` (wall
time, `max_memory_allocated`). The native preprocessed nights are cached in
`baselines/cache/cbramod/native[_quick]/<subject>.npy`, at 6 × 200 Hz, float32 µV, about
140 MB per night.

| File | What |
|---|---|
| `model.py` | Our port of `upstream/models/cbramod.py`, `criss_cross_transformer.py` and `model_for_isruc.py` (`PatchEmbedding`, `CrissCrossLayer`, `CrissCrossEncoder`, `CBraMod`, `SequenceModel`), plus `stretch_patch_embedding`, `rescaled_kernel` and `nearest_neighbour_chain` |
| `runner.py` | `run(spec, settings, out_dir, quick)` dispatches on `protocol`. Native uses its own data and training loop. Ours calls `baselines.common.ours.run_ours` |
| `native.yaml`, `ours.yaml` | Every setting. `model.py` and `runner.py` hold no configuration values |

### Why a port and not `import models.cbramod`

Running the cloned upstream code was refused by this environment's permission system
("code from external"). The original instruction for this folder was also "read upstream,
do not execute it". So `model.py` re-implements the three upstream model files line for
line. It does not import `upstream/`, and `upstream/` is unchanged at `b9e9610`.

The port keeps everything that matters for equivalence:
- **Names.** Module and parameter names are upstream's, so the checkpoint loads with
  `load_state_dict(strict=True)` ("All keys matched").
- **Order.** Modules are built in upstream's order: one layer deep-copied 12 times, then
  `_weights_init`'s Kaiming-normal on every `nn.Linear`. A seeded run therefore draws the
  same random numbers before the head is built. This is **(inferred)**; it could not be
  checked against upstream without running upstream.
- **Forward pass.** It is upstream's: the pre-norm criss-cross block, S-attention on
  features `[:100]` and T-attention on `[100:]`, the dropouts, and the flatten head.
- **Changes.** The only additions are the `dropout` argument (upstream's default of 0.1,
  now set in YAML) and two hooks in `SequenceModel`: `channel_order` and `patch_samples`.
  Both are identity for native.

### native (decisions by the coordinator, implemented as listed)

- **Data.** Six derivations F3−TP10, C3−TP10, O1−TP10, F4−TP9, C4−TP9, O2−TP9, read in µV
  by `baselines.common.native.read_channels`.
  - Each derivation is resampled while it is read: `scipy.signal.resample_poly(x, 200,
    1000)`.
  - Then `mne.filter.filter_data(…, 0.3, 35, fir_design='firwin')`, which is the same
    filter as upstream's `raw.filter`, then `mne.filter.notch_filter(…, 60)` (ANPHY mains).
  - Then `/100`. There is no scaler.
- **Epochs.** The epochs are our pipeline's rows (`night_epochs`): every hypnogram row
  inside the recording, 30 s from its onset. Unscored rows are labelled −1 and excluded
  from the loss (`ignore_index=-1`) and from the scores.
- **Runs.** Each night is cut into non-overlapping runs of 20 epochs from its first row,
  plus one right-aligned run ending at the last row when the night is not a multiple of
  20. The same runs are used for train, val and test.
  - ISRUC drops those last epochs everywhere. We keep them so that every scored test epoch
    is predicted.
  - An epoch covered twice keeps the prediction from its non-overlapping run.
- **Model.** `SequenceModel` with 6 ch × 30 patches × 200 points: the pretrained backbone,
  `Linear(36000, 512)` + GELU, a 1-layer bidirectional transformer (d 512), and
  `Linear(512, 5)`. **26,471,261 parameters.**
- **Training.**
  - Seed 3407 (`setup_seed` as upstream).
  - AdamW with weight decay 5e-2. Parameters whose name contains `backbone` get lr 1e-4;
    all others get 5e-4 (= 0.001·√(64/256)).
  - `CosineAnnealingLR(T_max = 50 × steps per epoch, eta_min = 1e-6)`, stepped every
    iteration.
  - Gradient-norm clip 1, label smoothing 0.1, 50 epochs, batch 64 runs, shuffled.
  - Full fine-tuning, no warm-up, no early stopping.
  - After each epoch, val is scored on the scored epochs of the 5 val nights. The state
    with the best **validation κ** is kept (strict `>`, as upstream) and tested.
- **Micro-batches.** Batch 64 is processed as 2 micro-batches of 32 (`micro_batch_size`):
  the loss is summed over the labelled positions and divided by the batch's labelled
  count, so the gradient equals the full-batch gradient. CBraMod has only LayerNorm and
  GroupNorm, which are per-sample, so there are no batch statistics. Only the dropout
  draws differ. This halves the peak memory, from 47.1 to 23.7 GiB, because other jobs
  share the A100. Setting `micro_batch_size: 64` gives the literal single pass.
- **Reported metrics.** Besides the shared scores, `metrics.json` has
  `test_native_metrics`: upstream's pooled balanced accuracy, κ and weighted F1, plus ours.

### ours (decisions by the coordinator, implemented as listed)

- **Data and training.** Our protocol, unchanged, via `run_ours`: 83 EEG at 1 kHz, CAR,
  robust scaler clipped to ±20, runs `[B, 20, 83, 30000]`, our stage-2 trainer with a
  4-epoch frozen probe, `overrides: {}`, and seed 42 from the reference run for the
  network's initialisation.
- **What the frozen probe trains.** It trains only `classifier` (`Linear(512, 5)`).
  During those 4 epochs the randomly initialised `Linear(498000, 512)`, the sequence
  transformer and the stretched first convolution stay frozen, together with the
  pretrained backbone. So the probe is a linear read-out of a *random* projection of the
  pretrained features. That is what "exactly as Exp 15" implies here.
- **Patches: option (a).** 1 s patches of 1000 points, 30 per epoch.
  `stretch_patch_embedding` replaces only the first time-branch convolution with
  kernel 245, stride 125 and padding 122 (all ×5, centred). Its weights are the
  pretrained 49 taps linearly interpolated onto the 5× finer grid and divided by 5. The
  frequency branch keeps `Linear(101, 200)` and reads FFT bins 0–100 Hz of the 1000-point
  `norm='forward'` FFT. ACPE, the other conv blocks and all 12 transformer layers are
  pretrained.
- **Check of option (a)**, a scratch test on a synthetic 1–20 Hz signal compared with its
  `resample_poly` to 200 Hz through the pretrained encoder:

  | Compared output | Correlation | Relative error |
  |---|---|---|
  | time branch | 0.984 | – |
  | spectra | – | max abs diff 0.003 |
  | patch embeddings | 0.993 | 0.12 |
  | 12-layer encoder output | 0.986 | 0.17 |

  So the pretrained weights do see what they were trained on.
- **Head.** Upstream's, literally: flatten 83·30·200 = 498000 → `Linear(498000, 512)` +
  GELU, a 1-layer bidirectional transformer with no mask, and `Linear(512, 5)`.
  **263,020,961 parameters**, of which the flatten `Linear` is 254,976,512 and the
  backbone 4,889,500 (the stretched conv adds 4,900 and the 1000-point mask token 800).
- **Channel order.** The loader's montage order is not spatially local: consecutive
  indices are 8.5 cm apart on average on the co-registered positions, against a
  nearest-neighbour distance of 2.6 cm. So the network applies a fixed permutation, the
  `nearest_neighbour_chain`: start at the most anterior electrode (FPZ) and always step to
  the nearest unvisited one. That brings consecutive indices to 3.4 cm and lowers the mean
  pairwise distance inside a 19-index ACPE window from 10.2 to 8.4 cm (12.4 cm over the
  whole montage). A front-to-back sort was tried and rejected: its consecutive distance is
  9.2 cm. The order used is written to `metrics.json` (`channel_order`).
- **Causality.** Not changed, by decision: the sequence encoder is bidirectional. Our
  val/test items end at the scored epoch, so it attends only to that epoch and the 19
  before it. The exception is the first 19 epochs of a night, whose item is the night's
  first 20 epochs. In training, though, every position sees the whole run, including later
  epochs.

### Smoke tests, resources and timings (A100 80 GB, fp32)

Both quick runs finished and wrote `metrics.json`. Their outputs and caches were then
deleted.

| Run | Quick content | Wall time | Peak `max_memory_allocated` | Step time |
|---|---|---|---|---|
| `native --quick` | 2 train / 1 val / 1 test nights (all 886 scored epochs of EPCTL06 predicted and scored), 1 epoch of 2 steps of 64 runs | 3 min 4 s, of which about 40 s per night is the one-off preprocessing | 23.7 GiB (micro-batch 32) | 1.40 s per batch of 64 |
| `ours --quick` | 2 train / 1 val / 1 test nights, 48 val and 48 test items, 1 epoch (frozen probe only) | 1 min 47 s | 4.4 GiB (probe: backbone frozen) | about 1.0 s per step of 2 runs, including loading |
| ours, unfrozen (scratch run with `freeze_epochs: 0`, quick) | 51 steps of batch 2 | 46 s per epoch | **28.6 GiB** | 0.9 s per step |
| measurement script, ours, random input, AdamW, no EMA | batch 2, full step | – | 27.8 GiB | 0.50 s (the first step takes 47 s for cuDNN set-up) |
| measurement script, native, random input | batch 64, single pass | – | 47.1 GiB | 0.99 s |

So a full training step of `ours` at batch 2 fits in 80 GB (28.6 GiB), and the literal
263 M-parameter head was kept. Full-run estimates **(inferred)**:
- **native:** about 15 steps per epoch, roughly 0.5 min per epoch with validation, so about
  30 min for 50 epochs, plus about 20 min of preprocessing the first time.
- **ours:** about 425 training steps per epoch (about 6–7 min unfrozen), plus validation
  on about 4.5k items, about 10–15 min per epoch. That is several hours for up to 40
  epochs, with patience 10.

Quick scores are meaningless (2–4 steps) and are not reported.

## Port verified against upstream (2026-11-03)

`model.py` is a port of `upstream/models/{cbramod,criss_cross_transformer,model_for_isruc}.py`.
It was checked against the upstream classes, imported from `upstream/` at `b9e9610`, with the
user's approval to execute upstream code:

- `SequenceModel` (native sizes) and upstream `Model` have the same parameter names; the only
  extra is the port's `channel_order` buffer.
- Built after `torch.manual_seed(3407)`, both give bit-identical initial weights.
- With the pretrained weights, the maximum absolute logit difference on a random
  `[2, 20, 6, 6000]` input in eval mode is **0.0**.
