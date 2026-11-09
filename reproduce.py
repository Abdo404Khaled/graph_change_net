"""Run every experiment of the paper in bulk, or any group of them.

The groups and their commands are listed in ``reproduce.yaml``. Each step is
one command template; its ``for`` mapping expands it over every combination of
the listed values (first key outermost), and its optional ``done`` path marks a
finished command, which ``--resume`` skips. Commands run from the repository
root with the current Python interpreter, one after another.

Examples
--------
``python reproduce.py --list``
    the groups and how many commands each runs;
``python reproduce.py final rotation --dry-run``
    print the commands of two groups without running them;
``python reproduce.py all --resume``
    everything in the plan's order, skipping commands whose output exists.
"""
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "reproduce.yaml"


def expand(step: dict) -> list[tuple[str, Path | None]]:
    """Commands of one step, with the file that marks each one as finished.

    Parameters
    ----------
    step : dict
        ``command`` template, optional ``for`` mapping of placeholder to values
        and optional ``done`` path template.

    Returns
    -------
    list of (str, Path or None)
        One entry per combination of the ``for`` values, in order.
    """
    loops = step.get("for", {})
    commands = []
    for values in itertools.product(*loops.values()):
        fields = dict(zip(loops, values))
        done = step.get("done")
        commands.append((step["command"].format(**fields).strip(),
                         ROOT / done.format(**fields) if done else None))
    return commands


def commands_of(plan: dict, group: str) -> list[tuple[str, Path | None]]:
    """Every command of one group, in order.

    Parameters
    ----------
    plan : dict
        Parsed ``reproduce.yaml``.
    group : str
        Group name.

    Returns
    -------
    list of (str, Path or None)
        Commands and their completion markers.
    """
    return [command for step in plan["groups"][group]["steps"] for command in expand(step)]


def run(command: str) -> None:
    """Run one command from the repository root with this interpreter.

    Parameters
    ----------
    command : str
        Shell-style command; a leading ``python`` is replaced by ``sys.executable``.

    Raises
    ------
    subprocess.CalledProcessError
        If the command fails.
    """
    words = shlex.split(command)
    if words[0] == "python":
        words[0] = sys.executable
    subprocess.run(words, cwd=ROOT, check=True)


def main() -> None:
    """Command line: ``python reproduce.py <group ...|all> [--dry-run] [--resume] [--list]``."""
    plan = yaml.safe_load(PLAN.read_text())
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("groups", nargs="*", metavar="GROUP",
                        help=f"one or more of: all, {', '.join(plan['order'])}")
    parser.add_argument("--list", action="store_true", help="list the groups and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    parser.add_argument("--resume", action="store_true",
                        help="skip commands whose output already exists")
    args = parser.parse_args()
    if args.list or not args.groups:
        for name in plan["order"]:
            count = len(commands_of(plan, name))
            print(f"{name:20s} {count:4d} commands  {plan['groups'][name]['description']}")
        return
    groups = plan["order"] if args.groups == ["all"] else args.groups
    unknown = [name for name in groups if name not in plan["groups"]]
    if unknown:
        parser.error(f"unknown group(s) {unknown}; choose from all, {', '.join(plan['order'])}")
    for group in groups:
        print(f"## {group}: {plan['groups'][group]['description']}", flush=True)
        for command, done in commands_of(plan, group):
            if args.resume and done is not None and done.exists():
                print(f"done  {command}", flush=True)
                continue
            print(f"run   {command}", flush=True)
            if not args.dry_run:
                run(command)


if __name__ == "__main__":
    main()
