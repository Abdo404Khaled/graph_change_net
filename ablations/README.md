# Ablations of the graph hierarchy

GraphChangeNet learns graphs at two levels:

- **Electrode level:** `L^S + Δ_t`, one signed electrode graph per 1 s segment (30 per
  epoch), with `Δ_t` carried forward causally (`graph_learner: residual_change`).
- **Segment level:** a graph over the 30 segments of the window
  (`sequence: window_graph_transformer`).

Each ablation starts from Exp 15 (`configs/exp15_final.yaml`) and changes one level, so a
difference from Exp 15 can be attributed to that level. The published graph sleep models
(one electrode graph per 30 s epoch) are external baselines and live in `baselines/`; these
ablations are changes to our own model.

| Ablation | Change from Exp 15 | Level | Question | Parameters |
|---|---|---|---|---|
| `no_segment_graph` | `sequence: lag_transformer`, Exp 11a's control: the same blocks and lag bias without the segment graph | segment | Does the segment graph matter? | 1,827,279 |
| `stable_only` | `graph_learner: stable_only`, `A_t = L^S` (`Δ_t = 0`) | electrode | Does the change term matter? | 2,009,422 |
| `no_memory` | `graph_learner.memory: false`, each `Δ_t` from its own segment only | electrode | Does the causal history of `Δ` matter? | 2,025,422 |
| `input_change` | `graph_learner: input_change`, `delta_t` from the projected input segment instead of its residual under `L^S` (energy weight fixed to one) | electrode | Does building the change from the residual matter? | 2,025,423 |
| `distance_graph` | `graph_learner: distance_kernel`, the fixed thresholded Gaussian kernel of scalp distance of Tang et al. (ICLR 2022), nothing learned | electrode | A learned or a fixed electrode graph? | 2,006,766 |
| `epoch_graph` | `graph_learner: epoch_change`, one graph per 30 s window (`L^S` + the change averaged over its segments) | electrode | Does the 1 s granularity matter? Breaks causality inside the window on purpose, like every per-epoch model. | 2,025,422 |
| `no_electrode_graph` | `graph_learner: no_graph`, every edge zero; each electrode keeps its own path | electrode | Does the electrode graph matter at all? | 2,006,766 |
| `no_graph` | `no_electrode_graph` and `no_segment_graph` together | both | Lower bound with no graph learning (two changes on purpose) | 1,808,622 |

Final-round runs (2026-11-05), chosen from the fold-0 results; the first three are candidates
for the final model and run on all five folds, the rest on fold 0:

| Run | Change from Exp 15 | Question | Folds |
|---|---|---|---|
| `wideband` | `data.filter: highpass`, `data.notch_harmonics: 7`: 0.5 Hz high-pass and notches at 60-420 Hz, no low-pass | Does the band above 45 Hz (present in the recordings up to about 450 Hz) help staging? | 0-4 |
| `cluster_mask` | `pretrain.mask_mode: cluster`, `cluster_size: 4`: each hidden electrode takes its 4 nearest neighbours with it | Does forcing non-local spatial inference in stage 1 help? | 0-4 |
| `wideband_cluster` | both (two changes on purpose) | The combination | 0-4 |
| `raw` | `data.filter: none`: no notch, no band-pass | Can a model learn from unfiltered EEG? | 0 |
| `no_electrode_graph_no_stage1` | `no_electrode_graph` without stage 1, against Exp 15a | The electrode graph without the pretraining confound | 0 |
| `electrodes_19` | `data.channel_subset`: the 19 standard 10-20 electrodes | Does electrode density matter? | 0 |

`wideband`, `wideband_cluster`, `raw` and `electrodes_19` need a new cache; build it once per
cache before training (`scripts/abci_ablation_prepare.sh`; `wideband_cluster` shares
`wideband`'s).

The components are registered in `components.py` (AGENTS.md: a model change is a new
registered component; `residual_change` and every other existing component are unchanged).
Each configuration was checked against the configuration Exp 15 ran with: apart from its
name and output folder, it differs only in the component(s) listed above.

## Layout

```text
ablations/
  configs/<name>.yaml   one ablation each, base ../../configs/exp15_final.yaml
  run.py                python -m ablations.run <name> --fold <k> [--quick] [--no-plots]
  summary.py            python -m ablations.summary   (every ablation against Exp 15)
  results/              <name>_fold<k>/ written by run.py (not committed)
```

`ablations.run` calls our `run.py` with the configuration and `paths.runs_dir` set to
`ablations/results`, so data, stage 1, stage 2, evaluation, the shuffled-segment control and
the reconstruction report are exactly those of Exp 15. On ABCI:

```bash
qsub -P <group> -v ABLATION=<name>,FOLD=0 scripts/abci_ablation.sh      # one fold
qsub -P <group> -J 1-5 -v ABLATION=<name> scripts/abci_ablation.sh      # all five folds
```

Each run takes about as long as Exp 15 (about 27 GPU-hours per fold).

## Reporting

`python -m ablations.summary` prints, per ablation, κ and macro-F1 over the finished folds,
the paired difference to Exp 15 on the same folds and on how many folds it is lower, the
parameter count, the shuffled-segment control and the masked-electrode reconstruction
relative to the nearest-electrode baseline. One seed and five test subjects are noisy:
a difference of about 0.01-0.02 κ is not a result.
