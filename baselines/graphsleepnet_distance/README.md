# GraphSleepNet with a fixed distance graph

GraphSleepNet (see `../graphsleepnet/README.md` and its `paper.pdf`) with the learned electrode
graph replaced by a **fixed scalp-distance graph**. Jia et al. ran this ablation themselves
(IJCAI 2020, Fig. 4a: on MASS-SS3, a distance graph gave about 0.880 accuracy against 0.889 for
the learned one, read off the bar chart). It is the "static graph, distance only" baseline:
it differs from `graphsleepnet` in the graph and nothing else.

| | `native.yaml` | `ours.yaml` |
|---|---|---|
| Everything but the graph | as `graphsleepnet/native.yaml` | as `graphsleepnet/ours.yaml` |
| Graph | the benchmark's own distance graph for the 23 prepared channels (`models.mstgcn.load_or_generate_adjacency`, the static graph native MSTGCN uses): Gaussian of 2-D head-coordinate distance, σ = 0.35, rows normalised | the same kernel on our 3-D montage positions (scaled to a mean radius of one), because upstream's coordinate table knows few of our 83 electrodes |
| How | `native_launcher.py` runs upstream's own trainer with `GraphSleepNetDE.forward` passing the graph through upstream's `adjacency_override` (upstream refuses `--fixed-graph` for native runs) | the wrapper passes it through the same argument |

Upstream's coordinate rule places EOG channels by index parity, so in the native graph EOGL
sits on the right and EOGR on the left; this is upstream's graph and is kept as it is. The
graph is saved as `upstream/distance_adjacency.npy` in the run folder.
