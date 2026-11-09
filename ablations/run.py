from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
RESULTS = ROOT / "results"


def names() -> list[str]:
    """Every ablation, by the file name of its configuration.

    Returns
    -------
    list of str
        Stems of ``ablations/configs/*.yaml``, sorted.
    """
    return sorted(path.stem for path in (ROOT / "configs").glob("*.yaml"))


def command(name: str, fold: int, quick: bool, plots: bool) -> list[str]:
    """The ``run.py`` command line of one ablation on one fold.

    The ablation runs through our own pipeline, exactly as Exp 15 did
    (``run.py --only exp15_final --set split.fold=<k>``): its configuration
    is ``ablations/configs/<name>.yaml``, whose ``base`` chain starts at
    ``configs/exp15_final.yaml``, and only ``paths.runs_dir`` is moved to
    ``ablations/results`` so the runs stay in this folder. The run is named
    ``<name>_fold<k>`` (``_quick_fold<k>`` for smoke tests).

    Parameters
    ----------
    name : str
        Ablation name.
    fold : int
        Fold index.
    quick : bool
        Smoke test with ``configs/quick.yaml``.
    plots : bool
        Write the figures.

    Returns
    -------
    list of str
        Arguments for :func:`subprocess.run`.
    """
    arguments = [sys.executable, str(REPOSITORY / "run.py"),
                 "--config", str(ROOT / "configs" / f"{name}.yaml"),
                 "--set", f"paths.runs_dir={RESULTS.relative_to(REPOSITORY)}",
                 f"split.fold={fold}"]
    if quick:
        arguments.append("--quick")
    if not plots:
        arguments.append("--no-plots")
    return arguments


def main() -> None:
    """Command line: ``python -m ablations.run <name> [--fold K] [--quick] [--no-plots]``.

    Raises
    ------
    SystemExit
        With ``run.py``'s exit code when the run fails.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("name", choices=names())
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(command(args.name, args.fold, args.quick, not args.no_plots),
                            cwd=REPOSITORY)
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
