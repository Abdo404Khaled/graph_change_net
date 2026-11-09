"""Write the split of every run of a rotation as a fold file the baselines read.

Usage: ``python scripts/write_rotation_folds.py <experiment config> <output dir>``,
for example ``python scripts/write_rotation_folds.py exp15_rotation baselines/folds``.
For each test group ``k`` of the config's ``split.n_test_groups`` it writes
``<output dir>/rot<k>.json`` with the same train, validation and test nights
that ``run.py`` uses for ``split.test_group=k``, so a baseline trained on that
fold is scored on exactly the nights of the matching GraphChangeNet run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as config_module  # noqa: E402
import data  # noqa: E402


def write_folds(experiment: str, out_dir: Path) -> list[Path]:
    """Compute and write the split of every test group of a rotation.

    Parameters
    ----------
    experiment : str
        Experiment config name or path with ``split.n_test_groups`` set.
    out_dir : Path
        Folder that receives ``rot<k>.json``.

    Returns
    -------
    list of Path
        The files written, in group order.

    Raises
    ------
    ValueError
        If the config does not define a rotation, or the groups do not test
        every subject exactly once.
    """
    base = config_module.resolve(experiment)
    n_groups = base.split.n_test_groups
    if not n_groups:
        raise ValueError(f"{experiment} has no split.n_test_groups")
    arrays, _, _ = data.load_dataset(base)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, tested = [], []
    for group in range(n_groups):
        cfg = config_module.resolve(experiment, [{"split": {"test_group": group}}])
        split = data.split_subjects(cfg, arrays)
        tested += split["test"]
        fold = {"source": f"rotation of {experiment}", "experiment": cfg.experiment,
                "commit": commit, **split}
        path = out_dir / f"rot{group}.json"
        path.write_text(json.dumps(fold, indent=1) + "\n")
        paths.append(path)
        print(f"{path}: test {split['test']} val {split['val']} train {len(split['train'])}")
    if sorted(tested) != sorted(map(str, arrays)):
        raise ValueError("the rotation does not test every subject exactly once")
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    write_folds(sys.argv[1], Path(sys.argv[2]))
