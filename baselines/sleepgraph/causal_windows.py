"""Past-only context windows for upstream's sleep-graph models (causal control).

Upstream builds a window of ``window_size`` epochs centred on the scored epoch,
so GraphSleepNet and MSTGCN see 2 later epochs and ProductGraphSleepNet 4. The
functions here keep the window length and every other rule (segment-local
contexts, ``drop`` or ``edge`` padding) but end the window at the scored epoch,
so no later epoch reaches the model. :func:`install` replaces upstream's two
functions in ``data.recordings`` before any trainer imports them; the label of
a window is still that of the scored epoch, now its last position. Imported
only in processes whose ``sys.path`` starts at the upstream repository.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import data.recordings as recordings


def causal_window_centers(record, *, window_size: int, pad_mode: str) -> Iterable[int]:
    """Scored epochs whose past-only context stays inside one contiguous segment.

    Parameters
    ----------
    record : Recording
        One upstream recording.
    window_size : int
        Epochs per window (odd, as upstream requires).
    pad_mode : str
        ``drop`` keeps only epochs with ``window_size - 1`` earlier epochs in
        their segment; ``edge`` keeps every epoch.

    Returns
    -------
    iterable of int
        Scored epoch indices.
    """
    recordings._validate_window(window_size, pad_mode)
    segment_id = recordings._segment_ids(record)
    if not len(segment_id):
        return
    starts = np.r_[0, np.flatnonzero(segment_id[1:] != segment_id[:-1]) + 1]
    stops = np.r_[starts[1:], len(segment_id)]
    for start, stop in zip(starts.tolist(), stops.tolist()):
        yield from range(start + window_size - 1 if pad_mode == "drop" else start, stop)


def causal_context_indices(record, center: int, *, window_size: int, pad_mode: str) -> np.ndarray:
    """The ``window_size`` epochs ending at ``center``, padded only at a segment start.

    Parameters
    ----------
    record : Recording
        One upstream recording.
    center : int
        Scored epoch, the last position of the window.
    window_size : int
        Epochs per window.
    pad_mode : str
        ``drop`` or ``edge`` (the segment's first epoch repeated).

    Returns
    -------
    np.ndarray
        Epoch indices, oldest first.

    Raises
    ------
    IndexError
        If ``center`` is outside the recording, or a ``drop`` window starts
        before it.
    RuntimeError
        If the window would cross a timeline discontinuity.
    """
    recordings._validate_window(window_size, pad_mode)
    if center < 0 or center >= len(record.y):
        raise IndexError(f"Window end {center} is outside {Path(record.path).name}")
    segment_id = recordings._segment_ids(record)
    center_segment = int(segment_id[center])
    indices = np.arange(center - window_size + 1, center + 1, dtype=np.int64)
    if pad_mode == "drop":
        if indices[0] < 0:
            raise IndexError("Drop-mode window starts before the recording")
    else:
        segment_start = int(np.searchsorted(segment_id, center_segment, side="left"))
        indices = np.clip(indices, segment_start, center)
    if np.any(segment_id[indices] != center_segment):
        raise RuntimeError(f"{Path(record.path).name}: attempted to construct a context across "
                           "a timeline discontinuity")
    return indices


def install() -> None:
    """Replace upstream's centred window functions with the past-only ones."""
    recordings.iter_window_centers = causal_window_centers
    recordings.context_window_indices = causal_context_indices
