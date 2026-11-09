# GraphChangeNet: signed stable-plus-change electrode graphs for HD-EEG sleep staging

Sleep staging (W, N1, N2, N3, R) on [ANPHY-Sleep](https://osf.io/r26fh/), 28 nights of 83-electrode
EEG recorded at 1 kHz and never resampled. Every 1 s segment `t` of a 30 s epoch gets its own signed
electrode graph

```
A_t = L^S + Delta_t
```

where `L^S` is a stable graph learned from electrode embeddings and `Delta_t` is the change in
segment `t`, accumulated causally from the earlier segments. The model has three levels:

1. a signed graph convolution on `A_t` for each second;
2. a causal signed graph over the 30 seconds of the epoch, read by a causal Transformer;
3. a forward GRU over the last 20 epochs, then the classification head.

Training has two stages. Stage 1 trains levels 1–2 to reconstruct masked electrodes. Stage 2
trains the whole network for staging. The final model (`configs/exp15_final.yaml`) has
2,025,423 parameters.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # install torch for your CUDA version first
python scripts/download_anphy.py             # 86 GB of zips, about 132 GB unpacked, into data/anphy/
```

The first run filters every night (60 Hz notch, 0.5–45 Hz band-pass, average reference) and caches
it in `cache_v0/`. The whole-night cache takes about 133 GB. Cache names hash every preprocessing
setting, so changing one builds a new cache instead of reusing a stale one.

## Running one experiment

```bash
python run.py --list                                     # the experiments in experiments.yaml
python run.py --only exp15_final                         # one experiment (fold 0)
python run.py --only exp15_final --quick                 # pipeline check, a few minutes
python run.py --only exp15_final --set split.fold=3      # any setting can be overridden
python run.py --only exp15_final --print-config          # the fully resolved configuration
python run.py --config exp15_final --pretrained P.pt --checkpoint C.pt   # evaluate saved weights
```

Settings live in YAML. `configs/default.yaml` holds every setting. `configs/<experiment>.yaml`
inherits its parent through `base:` and states only what it changes. `experiments.yaml` lists
the runnable experiments, and `--set key=value` overrides anything. Each run writes
`runs/<name>[_fold<k>|_rot<g>]/` with the resolved `config.yaml`, `metrics.json` (κ, macro-F1,
per-stage F1, shuffled-segment control, reconstruction report, parameter count, split, commit),
`history.json`, the weights of both stages, and figures.

## Evaluation scopes

| Scope | Nights | How to select it |
|---|---|---|
| Single split | 18 train / 5 validation / 5 test | `split.fold=0` |
| Five folds | the same 5 test nights; training and validation rotate | `split.fold=0..4` |
| Rotation | 7 groups of 4 nights, every night tested once (20 train / 4 validation) | `exp15_rotation` with `split.test_group=0..6` |

Baselines use the same nights through the fold files in `baselines/folds/` (`fold0.json`,
`rot0..6.json`, which `scripts/write_rotation_folds.py` writes from the rotation config).

## Reproducing the paper

`reproduce.py` runs every experiment of the paper in bulk. The commands are listed in
`reproduce.yaml`, one group per part of the paper:

```bash
python reproduce.py --list                    # groups and their number of commands
python reproduce.py final rotation --dry-run  # print the commands of some groups
python reproduce.py all --resume              # everything, skipping commands whose output exists
```

| Group | What it runs |
|---|---|
| `prepare` | data download, the baselines' upstream code and weights, the preprocessed caches |
| `final` | GraphChangeNet on the five folds |
| `development` | the runs that built the final model, five folds each |
| `removals` | the final model without stage 1, epoch context or the staging task, five folds each |
| `rotation` | GraphChangeNet on the 7 rotation groups |
| `ablations` | mechanism ablations of the graph hierarchy (fold 0) and input settings (fold 0 or five folds) |
| `baselines` | published models on the single split: native, unified input, causal and shared-CNN configurations |
| `baselines_rotation` | published models on the 7 rotation groups |
| `analysis` | predictions, paired bootstrap statistics, graph and stage analyses, cost, the paper's tables and figures |

Order and dependencies:

- Run `prepare` first.
- `baselines` and `analysis` read `runs/exp15_final_fold0`.
- `exp14b_epoch_context` starts from the stage-1 weights of `exp14a_scored_epochs` on the same fold.
- `analysis` reads the outputs of every training group.

Every other command is independent, so on a cluster each command can be one job. The PBS scripts in
`scripts/abci_*.sh` are examples: one experiment per job and an array over folds or rotation groups.

One GraphChangeNet run takes 7.5–10 h on one H200. The paper uses 78 runs of our model and its
ablations, plus 115 baseline runs. The `analysis` outputs go to `analysis/results/`, and the
tables and figures go to `analysis/results/paper/`.

## Layout

```
run.py, train.py, evaluate.py      entry point, two-stage training, metrics and controls
model.py, components.py            GraphChangeNet and its registered components (one per slot)
data.py, config.py, plots.py       loading, caching and splits; config schema; figures
configs/                           default.yaml and one file per experiment
experiments.yaml                   runnable experiments
reproduce.py, reproduce.yaml       bulk reproduction of the paper
ablations/                         ablations of our model (python -m ablations.run <name> --fold <k>)
baselines/                         published models and their adapters (python -m baselines.run <model> <config>)
analysis/                          evaluation scripts behind the paper's numbers, tables and figures
scripts/                           data download, baseline setup, rotation folds, cluster job examples
```

[ablations/README.md](ablations/README.md) and [baselines/README.md](baselines/README.md) describe
each ablation and each baseline configuration.

## Adding an experiment

1. A model change is a new class in `components.py`, registered with `@register("<slot>", "<name>")`.
   Existing components are never changed, so earlier runs keep reproducing.
2. Write `configs/<name>.yaml` with `base:` set to its parent and only the changed settings.
3. Add it to `experiments.yaml` and run it; add it to `reproduce.yaml` if it belongs to the paper.

The pipeline keeps these invariants: splits are by night, the scaler is fitted on training nights
only, masking happens before the graph is built, segment `t` sees only segments `1..t`, and the
1 kHz recordings are never resampled.
