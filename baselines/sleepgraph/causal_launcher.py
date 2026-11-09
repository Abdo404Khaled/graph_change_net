"""Upstream's training with past-only windows (causal control of the native graph models).

Run like ``python -m training.train`` (same arguments), in a process whose
``sys.path`` starts at the upstream repository. Installs the past-only window
functions of :mod:`causal_windows` before upstream's trainers are imported,
then hands over to upstream's ``training.train`` unchanged: the same
preparation, features, model, trainer and selection, but no scored epoch is
given any later epoch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent))

import causal_windows  # noqa: E402


def main() -> None:
    """Install the causal windows and run upstream's trainer with the given arguments."""
    causal_windows.install()
    from training import train
    train.main(sys.argv[1:])


if __name__ == "__main__":
    main()
