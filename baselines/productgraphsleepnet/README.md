# ProductGraphSleepNet

Einizade, Nasiri, Hajipour Sardouie, Clifford, "ProductGraphSleepNet: Sleep staging using
product spatio-temporal graph learning with attentive temporal aggregation", Neural Networks
164:667-680, 2023 (arXiv 2212.04881, `paper.pdf`).

**Method.** The same 9-band differential-entropy node features as GraphSleepNet, over a
window of 9 epochs. It learns **two graphs per window**: a spatial graph over the electrodes
and a temporal graph over the window's epochs, combined as a product graph in a Chebyshev
convolution, followed by a bidirectional GRU and a graph-wise attention that produces the
logits. Of the published graph sleep models it is the closest to our hierarchy (a graph over
electrodes and a graph over time), with one spatial graph per window rather than per second.

**Reported.** Accuracy 0.867 / 0.838, macro-F1 0.818 / 0.774, κ 0.802 / 0.775 on MASS-SS3 /
Sleep-EDF (abstract).

**Code.** sleep-graph-benchmark (`../sleepgraph/`).

| | `native.yaml` | `ours.yaml` |
|---|---|---|
| Data | upstream's ANPHY preparation: 19 EEG + 2 EOG + 2 EMG, 100 Hz | our 83 EEG at 1 kHz, our preprocessing and scaler |
| Node features | DE, z-scored on the training nights | the same DE from our epochs, z-scored on 100 unaugmented training runs |
| Window | 9 centred | 9 ending at the labelled epoch |
| Training | upstream: Adam 1e-3, batch 64, cross-entropy + the graph-smoothness loss | our stage 2 from scratch: classification loss only |
| Parameters | recorded in `metrics.json` (23 nodes) | 1,735,297 (83 nodes) |
