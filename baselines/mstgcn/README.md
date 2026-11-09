# MSTGCN

Jia, Lin, Wang, Zhou, Liu, Wang, Feng, "Multi-view spatial-temporal graph convolutional
networks with domain generalization for sleep stage classification", IEEE TNSRE 29:1977-1986,
2021 (arXiv 2109.01824, `paper.pdf`).

**Method.** A two-branch CNN (FeatureNet, DeepSleepNet-like) turns each channel's 30 s epoch
into 256 features. A window of 5 epochs goes through two graph views: a **learned functional
graph per epoch** and a **fixed distance graph** (spatial proximity), each with spatial and
temporal attention, Chebyshev graph convolution and temporal convolution; their outputs are
concatenated. A gradient-reversal domain head pushes the features to be subject-independent.
Of the published graph sleep models it is the closest to our "static + dynamic" idea.

**Reported.** MASS-SS3, 31-fold subject-independent CV: accuracy 0.895, macro-F1 0.854,
κ 0.843 (Table II); ISRUC-S3, 10-fold.

**Code.** sleep-graph-benchmark (`../sleepgraph/`), native = its two-stage `official` path.

| | `native.yaml` | `ours.yaml` |
|---|---|---|
| Data | upstream's ANPHY preparation: 19 EEG + 2 EOG + 2 EMG, 100 Hz | our 83 EEG at 1 kHz, our preprocessing and scaler |
| FeatureNet | official, trained first (stage 1), 3000-sample epochs only | `model.FeatureNet`: upstream's network with the first kernels and strides ×10 for 1 kHz (500/60, 4000/500), still 256 features per channel; identical to upstream at its own sizes (`check_equivalence` gives 0.0) |
| Graphs | learned per epoch + the benchmark's generated distance graph | learned per epoch + the same kernel (σ = 0.35) on our 3-D montage positions |
| Training | upstream: FeatureNet, then the graph network (Adam 1e-5) with graph losses and the domain head | one stage, our trainer from scratch: classification loss only, no domain head (replaced by an identity: 27M unused parameters) |
| Window | 5 centred | 5 ending at the labelled epoch |
| Parameters | recorded in `metrics.json` (23 nodes) | 28,270,849 (83 nodes; the class head alone is 27M: 2·128·5·83 inputs) |
