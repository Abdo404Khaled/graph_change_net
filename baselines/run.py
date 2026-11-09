from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIGS = ("native", "ours", "cnn2s", "causal")


def run(model: str, config: str, quick: bool, fold: Path | None = None) -> dict:
    """Run one baseline under one configuration and score it on the fold's test nights.

    ``baselines/<model>/<config>.yaml`` holds the configuration and
    ``baselines/<model>/runner.py`` its ``run(config, settings, out_dir, quick)``.
    Outputs go to ``<results_dir>/<model>/<config>`` (``_quick`` appended for
    smoke tests): the configuration used, the predictions and ``metrics.json``.
    A fold other than the one in ``settings.yaml`` (a run of a rotation,
    written by ``scripts/write_rotation_folds.py``) replaces it for this run,
    and the output folder gets ``_<fold file stem>`` appended so the runs of a
    rotation do not overwrite each other or the fold-0 results.

    Parameters
    ----------
    model : str
        Sub-package name, e.g. ``anysleep``.
    config : str
        ``native``, ``ours``, ``cnn2s`` (the graph models with the shared
        waveform CNN of Juvina & Nicolae, trained in two stages) or
        ``causal`` (the native configuration with past-only context).
    quick : bool
        Smoke test on a few nights, items and steps.
    fold : Path, optional
        Fold file to use instead of ``settings["fold"]``.

    Returns
    -------
    dict
        The run's scores.

    Raises
    ------
    ValueError
        If the configuration name is unknown.
    FileNotFoundError
        If ``fold`` is given and does not exist.
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, not {config!r}")
    settings = yaml.safe_load((ROOT / "settings.yaml").read_text())
    spec = yaml.safe_load((ROOT / model / f"{config}.yaml").read_text())
    suffix = ""
    if fold is not None:
        if not fold.exists():
            raise FileNotFoundError(f"fold file {fold} does not exist")
        settings["fold"] = str(fold)
        suffix = f"_{fold.stem}"
    out_dir = (Path(settings["results_dir"]) / model
               / (config + suffix + ("_quick" if quick else "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    runner = importlib.import_module(f"baselines.{model}.runner")
    return runner.run(spec, settings, out_dir, quick)


def main() -> None:
    """Command line: ``python -m baselines.run <model> <native|ours|cnn2s|causal> [--quick] [--fold FILE]``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model")
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fold", type=Path, default=None,
                        help="fold file to use instead of settings.yaml's (e.g. baselines/folds/rot3.json)")
    args = parser.parse_args()
    result = run(args.model, args.config, args.quick, args.fold)
    print(json.dumps({key: result[key] for key in ("n_epochs", "test", "parameters")
                      if key in result}, indent=1, default=float))


if __name__ == "__main__":
    main()
