# MTSSRL-MD (reference only, not runnable)

## Citation and links

- *MTSSRL-MD: Multi-Task Self-Supervised Representation Learning for EEG Signals across
  Multiple Datasets.* Anonymous authors, under review at ICLR 2026.
  OpenReview id `qD4N15aL2W`.
- **Code:** none released. The paper promises "an anonymous repository link during the
  discussion phase".
- **PDF:** supplied by the user in chat. Save it here as `paper.pdf`; OpenReview blocks
  automated download.

There is no code and no subject list, so this folder cannot be run. It is kept so that its
ANPHY numbers, the only published supervised baselines trained on ANPHY, are documented next
to the runnable baselines.

## Paper summary

- **Problem.** Sleep EEG datasets are small, sparsely labelled and use different montages.
  Single-dataset SSL generalises poorly, especially for N1 and REM.
- **Method.**
  - Datasets: SleepEDF-20, ISRUC-S1 and ANPHY-Sleep, pretrained jointly.
  - Channel Alignment Module (CAM): a spatial-attention layer per dataset, after Défossez
    et al. 2023, that maps each montage into 16 shared virtual channels. Positional
    dimension 72.
  - Shared encoder: a DeepSleepNet-style two-branch CNN with self-attention instead of the
    BiLSTM.
  - Three pretext tasks, weighted by Kendall uncertainty weighting:
    - augmentation contrastive (SimSiam, with SeqCLR augmentations);
    - temporal-shuffling discrimination (soft-margin loss);
    - frequency-band masking: classify which of 6 bands (δ 0.5–4, θ 4–8, α 8–12, σ 12–16,
      β 16–30, γ 30–50 Hz) was zeroed.
  - Fine-tuning: CAM frozen, attention pooling + MLP, progressive unfreezing,
    cross-entropy.
- **Training.** PyTorch 2.2.2, RTX 3090. AdamW, 400 pretraining epochs, batch 100,
  lr 1e-4.

## ANPHY setting (paper, section 4.1 and Appendix A)

| Item | Value |
|---|---|
| Subjects / epochs | "29 subjects", 16,526 epochs: W 3,322, N1 1,663, N2 6,564, N3 2,985, REM 1,992. Our pipeline counts 23,407 scored epochs on the 28 public nights; the dropped epochs are not explained |
| Channels | 11 EEG: C3, C4, Cz, O1, O2, P3, P4, Pz, F3, F4, Fpz (Fig. 2c, Fig. 3c). No EOG |
| Rate / filter | resampled to 100 Hz, band-pass 0.3–40 Hz, 30 s epochs |
| Split | subject-wise: 80% train / 20% test, with 20% of train held out for validation. Subject ids "in the released configuration files" (not released) |
| Labels | 5%, 10% or 100% of training epochs, stratified sampling |
| Metrics | balanced accuracy, macro-F1, Cohen's κ. The number of runs behind "±" is not stated |

## ANPHY results (paper, Tables 1–5; BACC / MF1 / κ in %)

| Model | 100% labels |
|---|---|
| DeepSleepNet (supervised) | 78.90 ± 0.18 / 77.34 ± 0.43 / 75.90 ± 0.36 |
| U-Sleep (supervised) | 77.20 ± 1.20 / 77.06 ± 0.50 / 74.72 ± 1.31 |
| ATCNet (supervised) | 77.80 ± 1.10 / 77.60 ± 0.53 / 75.00 ± 0.52 |
| **MTSSRL-MD** | **79.26 ± 0.78 / 79.20 ± 0.64 / 77.15 ± 0.85** |

| Model | 5% labels | 10% labels |
|---|---|---|
| MTSSRL-MD (S+I+A) | 68.02 / 67.76 / 66.04 | 74.33 / 74.52 / 70.22 |
| MTSSRL-MD, ANPHY-only pretraining | 64.32 / 63.12 / — | 70.60 / 70.03 / — |
| SeqCLR (S+I+A) | 64.31 / 63.77 / 62.50 | 69.24 / 68.80 / 66.23 |
| CPC pretrained on ANPHY | 64.73 / 64.71 / 64.08 | 71.09 / 70.91 / 68.09 |

Task ablation on ANPHY at 10% labels, BACC / MF1 (Table 4):

| Setting | BACC / MF1 |
|---|---|
| augmentation contrastive only | 72.61 / 72.31 |
| temporal shuffling only | 72.20 / 70.88 |
| band masking only | 74.50 / 73.94 |
| equal weights | 73.43 / 73.05 |
| uncertainty weighting | 74.33 / 74.52 |

## Points to note when citing

- The split differs from ours, and the subject ids are not public.
- The setting is 100 Hz with 11 EEG channels and no EOG.
- The epoch count is about 29% below ours.
- The paper claims "the first" framework of its kind.
- The efficiency table is inconsistent: DeepSleepNet is listed at 144.3M parameters, and
  MTSSRL-MD at 0.010M parameters but 4.87 GFLOPs.
