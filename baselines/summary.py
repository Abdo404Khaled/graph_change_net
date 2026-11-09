from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
COLUMNS = ("kappa", "macro_f1", "accuracy", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_R")


def rows(settings: dict) -> list[tuple[str, dict, int | None, int | None]]:
    """Every scored run: the reference model first, then each baseline result.

    A baseline result is any ``metrics.json`` under ``results_dir`` written by
    :func:`baselines.common.scoring.score_run`; its row is named by its path,
    so a model that saves several prediction files (one per checkpoint) gives
    several rows.

    Parameters
    ----------
    settings : dict
        Contents of ``baselines/settings.yaml``.

    Returns
    -------
    list of tuple
        ``(name, test scores, parameters, scored epochs)``.
    """
    reference = Path(settings["reference_run"])
    metrics = json.loads((reference / "metrics.json").read_text())
    found = [(f"reference: {metrics['experiment']}", metrics["test"], metrics.get("parameters"),
              None)]
    results = Path(settings["results_dir"])
    for path in sorted(results.rglob("metrics.json")):
        result = json.loads(path.read_text())
        if "test" not in result:
            continue
        name = str(path.parent.relative_to(results))
        if result.get("quick") or any(part.endswith("_quick") for part in path.parent.parts):
            name += " (quick)"
        found.append((name, result["test"], result.get("parameters"), result.get("n_epochs")))
    return found


def table(settings: dict) -> str:
    """The results as a Markdown table, kappa and macro-F1 first.

    Parameters
    ----------
    settings : dict
        Contents of ``baselines/settings.yaml``.

    Returns
    -------
    str
        Markdown table.
    """
    header = "| Run | Params | Epochs | " + " | ".join(COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 3)]
    for name, scores, parameters, n_epochs in rows(settings):
        values = " | ".join(f"{scores[key]:.4f}" for key in COLUMNS)
        size = f"{parameters:,}" if parameters else "—"
        lines.append(f"| {name} | {size} | {n_epochs or '—'} | {values} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(yaml.safe_load((ROOT / "settings.yaml").read_text())))
