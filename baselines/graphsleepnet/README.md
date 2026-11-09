# GraphSleepNet

Jia, Lin, Wang, Feng, Zhou, Liu, Wang, "GraphSleepNet: Adaptive spatial-temporal graph
convolutional networks for sleep stage classification", IJCAI 2020 (`paper.pdf`).

**Method.** Each 30 s epoch of each EEG channel becomes 9 differential-entropy features
(0.5-4, 2-6, 4-8, 6-11, 8-14, 11-22, 14-31, 22-40, 31-50 Hz). A window of 5 epochs is a
spatial-temporal graph: the **electrode graph is learned per window** from the centre epoch's
features (`A = softmax(|x_i - x_j| a)`, row-stochastic), and a block of spatial and temporal
attention, Chebyshev graph convolution (order 3) and a temporal convolution labels the centre
epoch. One electrode graph per 30 s epoch; time is modelled by convolution and attention over
epochs, not by a graph.

**Reported.** MASS-SS3, 20 EEG channels, 31-fold subject-independent CV: accuracy 0.889,
macro-F1 0.841, κ 0.834 (as listed in the MSTGCN paper, Table II).

**Code.** sleep-graph-benchmark's re-implementation of the released code (`../sleepgraph/`,
its README lists the native ANPHY settings); nothing in it is changed.

| | `native.yaml` | `ours.yaml` |
|---|---|---|
| Data | upstream's ANPHY preparation: 19 EEG + 2 EOG + 2 EMG, 0.3-100 Hz, 100 Hz | our 83 EEG at 1 kHz, our preprocessing and training-fitted scaler |
| Node features | DE, z-scored on the training nights | the same DE from our epochs, z-scored on 100 unaugmented random training runs |
| Window | 5 epochs centred on the labelled epoch | 5 epochs ending at the labelled epoch (causal) |
| Graph | learned from the centre epoch | learned from the labelled epoch |
| Training | upstream: Adam 1e-3, batch 64, up to 200 epochs, smoothed validation macro-F1 | our stage 2 from scratch (no frozen probe) |
| Parameters | recorded in `metrics.json` (23 nodes) | 81,916 (83 nodes) |

Every scored test epoch is predicted (`../sleepgraph/native_predict.py` re-runs the selected
checkpoint with edge-padded windows; where upstream also predicted, the predictions must
agree). `graphs.npz` keeps the electrode graph of every scored test epoch for the graph tests.
