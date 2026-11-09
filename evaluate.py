from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config


@torch.no_grad()
def predict(cfg: Config, model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted stages of every window in a loader.

    Parameters
    ----------
    cfg : Config
        Run settings (device).
    model : nn.Module
        Trained network.
    loader : DataLoader
        Windows to classify.

    Returns
    -------
    true : np.ndarray
        Stage indices from the hypnogram.
    predicted : np.ndarray
        Stage indices from the model.
    """
    true, probability = predict_proba(cfg, model, loader)
    return true, probability.argmax(-1)


@torch.no_grad()
def predict_proba(cfg: Config, model: nn.Module, loader: DataLoader,
                  zero_elapsed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """True stages and predicted stage probabilities of every window in a loader.

    Rolling windows (Exp 13) come with their elapsed time, which is passed to
    the model. Runs of epochs (Exp 14b) give one prediction per position and
    keep those labelled 0 or above, which in validation and test is the one
    epoch each item ends at.

    Parameters
    ----------
    cfg : Config
        Run settings (device).
    model : nn.Module
        Trained network.
    loader : DataLoader
        Windows to classify.
    zero_elapsed : bool
        Replace the elapsed time by 0, the training mean after
        standardisation, to measure how much the model relies on it.

    Returns
    -------
    true : np.ndarray
        Stage indices from the hypnogram.
    probability : np.ndarray
        Softmax outputs ``[n, n_stages]``.
    """
    model.eval()
    true, probability = [], []
    for x, y, *extra in loader:
        extra = [torch.zeros_like(item) if zero_elapsed else item for item in extra]
        logits = model(x.to(cfg.device), *[item.to(cfg.device) for item in extra])[0]
        labels = y.flatten().numpy()
        labelled = labels >= 0
        true.append(labels[labelled])
        probability.append(logits.flatten(0, -2).softmax(-1).cpu().numpy()[labelled])
    return np.concatenate(true), np.concatenate(probability)


@torch.no_grad()
def context_report(cfg: Config, model: nn.Module, loader: DataLoader,
                   test: dict[str, float]) -> dict:
    """Does the classifier use the earlier epochs, and their order (Exp 14b)?

    Two controls on the test items, each scored like the test:

    - ``shuffled_context``: the epochs before the scored one are put in a
      random order (the scored epoch stays last), averaged over
      ``evaluation.shuffle_trials`` draws; a drop means the order of the
      past matters, not only its content.
    - ``no_context``: each scored epoch is classified alone, as a run of one;
      a drop means the earlier epochs matter at all.

    Parameters
    ----------
    cfg : Config
        Run settings (device, ``evaluation.shuffle_trials``).
    model : nn.Module
        Trained network with a ``context`` component.
    loader : DataLoader
        Test items: runs of epochs labelled at their last scored position.
    test : dict
        Test metrics of the unchanged items.

    Returns
    -------
    dict
        Metrics of each control, each with ``delta_kappa`` and
        ``delta_macro_f1`` against ``test``.
    """
    model.eval()
    device = cfg.device
    generator = torch.Generator().manual_seed(0)
    report = {}
    for name, trials in (("shuffled_context", cfg.evaluation.shuffle_trials), ("no_context", 1)):
        collected = []
        for _ in range(trials):
            true, predicted = [], []
            for x, y in loader:
                target = (y >= 0).float().argmax(1)
                if name == "no_context":
                    x = x[torch.arange(len(x)), target].unsqueeze(1)
                    y = y[torch.arange(len(y)), target].unsqueeze(1)
                else:
                    order = torch.arange(x.shape[1]).repeat(len(x), 1)
                    for item, end in enumerate(target.tolist()):
                        order[item, :end] = order[item, torch.randperm(end, generator=generator)]
                    x = x[torch.arange(len(x)).unsqueeze(1), order]
                    y = y[torch.arange(len(y)).unsqueeze(1), order]
                logits = model(x.to(device))[0].flatten(0, -2)
                labels = y.flatten()
                labelled = labels >= 0
                true.append(labels[labelled].numpy())
                predicted.append(logits.argmax(-1).cpu().numpy()[labelled.numpy()])
            collected.append(scores(cfg, np.concatenate(true), np.concatenate(predicted)))
        result = {key: float(np.mean([row[key] for row in collected])) for key in collected[0]}
        result["delta_kappa"] = result["kappa"] - test["kappa"]
        result["delta_macro_f1"] = result["macro_f1"] - test["macro_f1"]
        report[name] = result
    return report


def scores(cfg: Config, true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Sleep-staging metrics: kappa and macro-F1 first, then accuracy and per-stage F1.

    ``macro_f1_present`` averages over the stages that occur only, because a
    split without a stage would otherwise average in an F1 of 0 for it.

    Parameters
    ----------
    cfg : Config
        Run settings (stage names).
    true, predicted : np.ndarray
        Stage indices.

    Returns
    -------
    dict
        ``kappa``, ``macro_f1``, ``accuracy``, ``macro_f1_present`` and ``f1_<stage>``.
    """
    stages = cfg.data.stages
    per_class = f1_score(true, predicted, average=None, labels=range(len(stages)),
                         zero_division=0)
    return {"kappa": float(cohen_kappa_score(true, predicted)),
            "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(true, predicted)),
            "macro_f1_present": float(f1_score(true, predicted, average="macro",
                                               labels=np.unique(true), zero_division=0)),
            **{f"f1_{stage}": float(score) for stage, score in zip(stages, per_class)}}


def evaluate(cfg: Config, model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Score a model on a loader.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : nn.Module
        Trained network.
    loader : DataLoader
        Windows to score.

    Returns
    -------
    dict
        The metrics of :func:`scores`.
    """
    return scores(cfg, *predict(cfg, model, loader))


@torch.no_grad()
def curve_scores(cfg: Config, model: nn.Module, loader: DataLoader,
                 batches: int) -> dict[str, float]:
    """Scores and plain cross entropy, computed the same way for training and validation epochs.

    The training loss of stage 2 is taken with dropout, augmentation, label
    smoothing, focal modulation and class weights, while the network
    changes over the epoch, so it cannot be set against a validation score.
    This scores a fixed set of epochs in eval mode with the unweighted cross
    entropy (the mean negative log-likelihood of the true stage), so the
    same function applied to a training subset and to the validation set
    gives two curves on one scale.

    Parameters
    ----------
    cfg : Config
        Run settings (device, stage names).
    model : nn.Module
        Network to score.
    loader : DataLoader
        Epochs in a fixed order.
    batches : int
        Batches to read; 0 reads the whole loader.

    Returns
    -------
    dict
        The metrics of :func:`scores` and ``cross_entropy``.
    """
    model.eval()
    true, predicted, total, count = [], [], 0.0, 0
    for index, (x, y, *extra) in enumerate(loader):
        if batches and index >= batches:
            break
        logits = model(x.to(cfg.device), *[item.to(cfg.device) for item in extra])[0]
        logits, labels = logits.flatten(0, -2), y.to(cfg.device).flatten()
        labelled = labels >= 0
        if not labelled.any():
            continue
        total += F.cross_entropy(logits[labelled], labels[labelled], reduction="sum").item()
        count += int(labelled.sum())
        true.append(labels[labelled].cpu().numpy())
        predicted.append(logits[labelled].argmax(-1).cpu().numpy())
    return {**scores(cfg, np.concatenate(true), np.concatenate(predicted)),
            "cross_entropy": total / max(count, 1)}


def nearest_visible(target: torch.Tensor, mask: torch.Tensor,
                    neighbours: np.ndarray) -> torch.Tensor:
    """Copy each hidden electrode from the closest electrode that is still visible.

    A strong baseline that needs no graph: neighbouring electrodes are highly
    correlated.

    Parameters
    ----------
    target : torch.Tensor
        True segments ``[B, T, C, L]``.
    mask : torch.Tensor
        ``[B, T, C]``, true where hidden.
    neighbours : np.ndarray
        ``[C, C - 1]`` electrode indices ordered by distance.

    Returns
    -------
    torch.Tensor
        Predicted segments, same shape as ``target``.
    """
    order = torch.as_tensor(neighbours, device=target.device)
    source = torch.full(mask.shape, -1, dtype=torch.long, device=target.device)
    for rank in range(order.shape[1]):
        candidate = order[:, rank].expand_as(source)
        source = torch.where(~torch.gather(mask, 2, candidate) & (source < 0),
                             candidate, source)
    index = source.clamp(min=0).unsqueeze(-1).expand_as(target)
    return torch.gather(target, 2, index)


@torch.no_grad()
def reconstruction_report(cfg: Config, model: nn.Module, loader: DataLoader,
                          coords: np.ndarray) -> list[dict]:
    """Score the reconstruction against three baselines on both masking regimes.

    Pass the pretrained weights: stage 2 moves the backbone but not the
    decoder. Random cells can be filled in from the same electrode over time,
    while a whole hidden electrode can only come from other electrodes, so the
    regimes are reported separately. The ``nearest_electrode`` baseline looks
    for a visible electrode among the ``cfg.evaluation.nearest_candidates``
    nearest ones.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : nn.Module
        Network with its stage 1 weights.
    loader : DataLoader
        Validation windows.
    coords : np.ndarray
        Electrode coordinates ``[C, 3]``.

    Returns
    -------
    list of dict
        One row per regime and predictor with ``loss``, ``variance_explained``
        and ``correlation``.
    """
    from train import sample_mask

    device = cfg.device
    candidates = min(cfg.evaluation.nearest_candidates, len(coords) - 1)
    neighbours = NearestNeighbors(n_neighbors=candidates + 1).fit(coords).kneighbors(
        coords, return_distance=False)[:, 1:]
    model.eval()
    torch.manual_seed(0)
    rows = []

    for regime, channel_prob in [("random_cells", 0.0), ("whole_electrodes", 1.0)]:
        totals: dict[str, list] = {}
        for index, (x, *_) in enumerate(loader):
            x = x.to(device)
            target = model.segments(x)
            mask = sample_mask(cfg, x.shape[0], x.shape[1], device, channel_prob)
            seen = (~mask).unsqueeze(-1)

            guesses = {"model": model.reconstruct(x, mask)[0],
                       "zeros": torch.zeros_like(target),
                       "visible_mean": ((target * seen).sum(2) / seen.sum(2).clamp(min=1)
                                        ).unsqueeze(2).expand_as(target),
                       "nearest_electrode": nearest_visible(target, mask, neighbours)}

            truth = target[mask]
            for name, guess in guesses.items():
                totals.setdefault(name, []).append((guess[mask], truth))
            if index + 1 >= cfg.evaluation.reconstruction_batches:
                break

        for name, pairs in totals.items():
            guess = torch.cat([g for g, _ in pairs])
            truth = torch.cat([t for _, t in pairs])
            rows.append({
                "regime": regime, "predictor": name,
                "loss": F.smooth_l1_loss(guess, truth).item(),
                "variance_explained": 1 - (((guess - truth) ** 2).sum()
                                           / ((truth - truth.mean()) ** 2).sum()).item(),
                "correlation": F.cosine_similarity(guess - guess.mean(-1, keepdim=True),
                                                   truth - truth.mean(-1, keepdim=True),
                                                   dim=-1).mean().item()})
    return rows


def format_reconstruction(rows: list[dict]) -> str:
    """Lay out the reconstruction report as a fixed-width table.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`reconstruction_report`.

    Returns
    -------
    str
        The table.
    """
    header = f"{'regime':<18}{'predictor':<20}{'loss':>9}{'var.expl':>10}{'corr':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        if row["predictor"] == "model":
            lines.append("-" * len(header))
        lines.append(f"{row['regime']:<18}{row['predictor']:<20}{row['loss']:>9.4f}"
                     f"{row['variance_explained']:>10.4f}{row['correlation']:>9.4f}")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_shuffled(cfg: Config, model: nn.Module, loader: DataLoader) -> tuple[dict, dict]:
    """Score the model with the segments of every window permuted.

    Same weights and signal, only the order changes, so a drop measures how
    much the model relies on segment order. Runs of epochs (Exp 14b+) have
    the segments of every epoch permuted the same way, and only their
    labelled positions are scored.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.evaluation.shuffle_trials``).
    model : nn.Module
        Trained network.
    loader : DataLoader
        Windows to score.

    Returns
    -------
    mean : dict
        Mean ``kappa``, ``macro_f1`` and ``accuracy`` over the permutations.
    std : dict
        Their standard deviations.
    """
    model.eval()
    n_segments = cfg.data.n_segments
    collected = []
    for trial in tqdm(range(cfg.evaluation.shuffle_trials), desc="shuffle trials"):
        generator = torch.Generator().manual_seed(trial)
        true, predicted = [], []
        for x, y, *extra in loader:
            cells = x.unflatten(-1, (n_segments, -1))
            order = torch.randperm(n_segments, generator=generator)
            shuffled = cells[..., order, :].flatten(-2)
            extra = [item.to(cfg.device) for item in extra]
            logits = model(shuffled.to(cfg.device), *extra)[0].flatten(0, -2)
            labels = y.flatten().numpy()
            labelled = labels >= 0
            predicted.append(logits.argmax(-1).cpu().numpy()[labelled])
            true.append(labels[labelled])

        true, predicted = np.concatenate(true), np.concatenate(predicted)
        collected.append({"kappa": cohen_kappa_score(true, predicted),
                          "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
                          "accuracy": accuracy_score(true, predicted)})

    keys = collected[0]
    return ({k: float(np.mean([c[k] for c in collected])) for k in keys},
            {k: float(np.std([c[k] for c in collected])) for k in keys})


@torch.no_grad()
def segment_graph_report(cfg: Config, model: nn.Module, loader: DataLoader,
                         test: dict[str, float]) -> dict:
    """How the learned segment graph splits into signs, and whether its edges matter.

    The edge share counts the sign of every pair of a segment and an earlier
    one, over all windows: the sampled type for discrete edges (Exp 8), the
    sign of the weight for continuous edges (Exp 9a+), which also report the
    mean absolute weight. The random-graph control then keeps the trained
    weights but places the edges at random: discrete types drawn with those
    shares, or each window's continuous weights shuffled over its pairs. If
    the scores hold, the learned graph is not what the model relies on (Cini
    et al., JMLR 2023, found random graphs often competitive). The global
    random state is restored.

    Parameters
    ----------
    cfg : Config
        Run settings (device, ``evaluation.shuffle_trials``).
    model : nn.Module
        Trained network whose sequence component learns a segment graph.
    loader : DataLoader
        Windows to score.
    test : dict
        Scores of the model with its learned graph, from :func:`scores`.

    For a component that only biases attention with the graph
    (``graph_bias``), the model is also scored with the bias removed, and the
    learned bias weights of every block are reported.

    Returns
    -------
    dict
        ``edge_share`` per edge type and ``random_graph`` with the mean and
        standard deviation over the trials and the drop in kappa and macro-F1;
        with a graph bias also ``no_graph_bias`` (scores and drops) and
        ``graph_weights`` (per block, positive and negative weight per head)
        when the blocks have them. A component with a lag graph (Exp 11) or a
        computed similarity graph (Exp 12) also reports :func:`graph_variation`;
        for these, ``no_graph_bias`` means the graph branch removed from every
        block. Exp 11 adds its learned ``lag_graph``. Exp 12 adds
        ``raw_similarity`` (mean and within-row spread of the similarity
        before standardisation; a spread near 0 means the segments look alike
        and the graph only amplifies noise) and ``mean_graph``: the scores
        with every window's graph replaced by the test-set average graph.
    """
    graph = model.sequence
    continuous = hasattr(graph, "edge_weights")
    lag_graph = hasattr(graph, "lag_graph")
    similarity = hasattr(graph, "similarity")
    variation = lag_graph or similarity
    model.eval()
    counts = torch.zeros(2, dtype=torch.float64)
    magnitude, pairs = 0.0, 0
    collected_weights, collected_labels = [], []
    graph_sum, windows = 0.0, 0
    raw_mean, raw_spread, raw_rows = 0.0, 0.0, 0
    for x, y, *_ in loader:
        hidden, _, _ = model.backbone(x.to(cfg.device))
        nodes = graph_input(graph, hidden)
        if continuous:
            full = graph.edge_weights(nodes)
            weights = full[:, graph.causal]
            counts += torch.stack([(weights > 0).sum(), (weights < 0).sum()]).double().cpu()
            magnitude += weights.abs().sum().item()
            pairs += weights.numel()
            if variation:
                collected_weights.append(weights.double().cpu())
                collected_labels.append(y)
            if similarity:
                graph_sum = graph_sum + full.double().sum(0)
                windows += len(full)
                raw = graph.similarity(nodes).double()
                causal = graph.causal.double()
                rows = causal.sum(-1) >= 2
                count = causal.sum(-1).clamp(min=1)
                mean = (raw * causal).sum(-1) / count
                spread = ((((raw - mean.unsqueeze(-1)) * causal) ** 2).sum(-1) / count).sqrt()
                raw_mean += mean[:, rows].sum().item()
                raw_spread += spread[:, rows].sum().item()
                raw_rows += mean[:, rows].numel()
        else:
            counts += graph.edge_types(nodes)[:, graph.causal].sum((0, 1)).double().cpu()
    share = counts / counts.sum()

    devices = [cfg.device] if cfg.device.type == "cuda" else []
    collected = []
    with torch.random.fork_rng(devices=devices):
        if continuous:
            graph.shuffle_edges = True
        else:
            graph.random_edges = share.float()
        try:
            for trial in range(cfg.evaluation.shuffle_trials):
                torch.manual_seed(trial)
                result = scores(cfg, *predict(cfg, model, loader))
                collected.append({key: result[key] for key in ("kappa", "macro_f1", "accuracy")})
        finally:
            if continuous:
                graph.shuffle_edges = False
            else:
                graph.random_edges = None

    mean = {k: float(np.mean([c[k] for c in collected])) for k in collected[0]}
    std = {k: float(np.std([c[k] for c in collected])) for k in collected[0]}
    report = {"edge_share": dict(zip(("positive", "negative"), share.tolist())),
              "random_graph": {"mean": mean, "std": std,
                               "delta_kappa": mean["kappa"] - test["kappa"],
                               "delta_macro_f1": mean["macro_f1"] - test["macro_f1"]}}
    if hasattr(graph, "graph_bias"):
        graph.graph_bias = False
        try:
            result = scores(cfg, *predict(cfg, model, loader))
        finally:
            graph.graph_bias = True
        report["no_graph_bias"] = {"kappa": result["kappa"], "macro_f1": result["macro_f1"],
                                   "delta_kappa": result["kappa"] - test["kappa"],
                                   "delta_macro_f1": result["macro_f1"] - test["macro_f1"]}
        if all(hasattr(block, "graph_weight") for block in graph.blocks):
            report["graph_weights"] = [block.graph_weight.tolist() for block in graph.blocks]
    if continuous:
        report["mean_abs_weight"] = magnitude / pairs
    if variation:
        report.update(graph_variation(torch.cat(collected_weights), torch.cat(collected_labels),
                                      cfg.evaluation.shuffle_trials))
    if lag_graph:
        report["lag_graph"] = graph.lag_graph.detach().cpu().tolist()
    if similarity:
        report["raw_similarity"] = {"mean": raw_mean / raw_rows, "spread": raw_spread / raw_rows}
        graph.mean_graph = (graph_sum / windows).float()
        try:
            result = scores(cfg, *predict(cfg, model, loader))
        finally:
            graph.mean_graph = None
        report["mean_graph"] = {"kappa": result["kappa"], "macro_f1": result["macro_f1"],
                                "delta_kappa": result["kappa"] - test["kappa"],
                                "delta_macro_f1": result["macro_f1"] - test["macro_f1"]}
    return report


def graph_input(graph: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """What a segment-graph component builds its graph from.

    Exp 8-11 build it from the segment tokens; Exp 12 (``graph_from_electrodes``)
    from the layer-1 electrode features themselves.

    Parameters
    ----------
    graph : nn.Module
        The sequence component.
    hidden : torch.Tensor
        Layer-1 node representations ``[B, T, C, D]``.

    Returns
    -------
    torch.Tensor
        ``hidden`` itself, or the tokens ``[B, T, hidden]``.
    """
    if getattr(graph, "graph_from_electrodes", False):
        return hidden
    return graph.embed(hidden.flatten(2))


def median_stage_f(values: torch.Tensor, labels: torch.Tensor) -> float:
    """Median over edges of the one-way ANOVA F statistic of edge weight by stage.

    Parameters
    ----------
    values : torch.Tensor
        Edge weights ``[N, P]``: ``N`` windows, ``P`` edges.
    labels : torch.Tensor
        Stage of every window ``[N]``.

    Returns
    -------
    float
        Median of the ``P`` F statistics; about 1 when the weights do not
        depend on the stage.
    """
    stages = labels.unique()
    grand = values.mean(0)
    between = sum((labels == stage).sum() * (values[labels == stage].mean(0) - grand) ** 2
                  for stage in stages) / (len(stages) - 1)
    within = sum(((values[labels == stage] - values[labels == stage].mean(0)) ** 2).sum(0)
                 for stage in stages) / (len(values) - len(stages))
    return float((between / within.clamp(min=1e-12)).median())


def graph_variation(values: torch.Tensor, labels: torch.Tensor, permutations: int) -> dict:
    """Whether a learned segment graph depends on the window and on the stage.

    Exp 10's graph was 95% the same matrix in every window and barely
    differed between stages, which the random-graph control alone cannot
    show. The share of the edge-weight variance that lies between windows
    (rather than between edges) is 0 for a fixed graph. The stage test
    compares the median F statistic of the edges with the same statistic
    after shuffling the stage labels; a private generator keeps the global
    random state untouched.

    Parameters
    ----------
    values : torch.Tensor
        Edge weights ``[N, P]`` of ``N`` windows over the ``P`` earlier ->
        later pairs.
    labels : torch.Tensor
        Stage of every window ``[N]``.
    permutations : int
        Label shuffles averaged for the reference statistic.

    Returns
    -------
    dict
        ``window_varying_share``, ``stage_f`` and ``stage_f_shuffled``.
    """
    generator = torch.Generator().manual_seed(0)
    shuffled = [median_stage_f(values, labels[torch.randperm(len(labels), generator=generator)])
                for _ in range(permutations)]
    return {"window_varying_share": float(values.var(0).mean() / values.var()),
            "stage_f": median_stage_f(values, labels),
            "stage_f_shuffled": float(np.mean(shuffled))}


@torch.no_grad()
def collect_segment_graphs(cfg: Config, model: nn.Module, loader: DataLoader) -> dict:
    """Average the learned segment graph per sleep stage, for the figures.

    Importance is shown relative to equal attention: segment ``t`` has ``t``
    earlier segments, so its importances are multiplied by ``t``, and 1 means
    an earlier segment gets an even share. Without this, early rows would look
    strong only because they have fewer segments to share among.

    Parameters
    ----------
    cfg : Config
        Run settings (device, stage names).
    model : nn.Module
        Trained network whose sequence component learns a segment graph.
    loader : DataLoader
        Windows to run.

    Returns
    -------
    dict
        ``groups`` (``"all"`` and every stage present), and per group the mean
        ``sign`` (probability of a positive edge for discrete edges, the signed
        weight for continuous ones, named by ``edge_label`` with lower bound
        ``edge_low``), ``positive`` and ``negative`` relative importance
        ``[T, T]`` (NaN where ``s >= t``), ``counts`` of windows, and the same
        three matrices for the first window as ``example_*``.
    """
    graph = model.sequence
    continuous = hasattr(graph, "edge_weights")
    model.eval()
    n_groups = len(cfg.data.stages) + 1
    n_segments = cfg.data.n_segments
    sums = torch.zeros(3, n_groups, n_segments, n_segments, dtype=torch.float64)
    counts = torch.zeros(n_groups, dtype=torch.float64)
    scale = torch.arange(n_segments, dtype=torch.float64).clamp(min=1).view(-1, 1)
    example = None
    for x, y, *_ in loader:
        hidden, _, _ = model.backbone(x.to(cfg.device))
        nodes = graph_input(graph, hidden)
        if continuous:
            sign = graph.edge_weights(nodes)
        else:
            sign = graph.edge_logits(nodes).softmax(-1)[..., 0]
        positive, negative = graph.importance(nodes)
        stacked = torch.stack([sign, positive, negative]).double().cpu()
        stacked[1:] = stacked[1:] * scale
        if example is None:
            example = stacked[:, 0].clone()
        groups = torch.cat([torch.zeros_like(y), y + 1])
        windows = torch.cat([torch.arange(len(y)), torch.arange(len(y))])
        for group in groups.unique():
            chosen = windows[groups == group]
            sums[:, group] += stacked[:, chosen].sum(1)
            counts[group] += len(chosen)

    causal = graph.causal.cpu()
    present = counts > 0
    means = sums[:, present] / counts[present].view(1, -1, 1, 1)
    means[:, :, ~causal] = float("nan")
    example[:, ~causal] = float("nan")
    names = ["all", *cfg.data.stages]
    return {"groups": [name for name, keep in zip(names, present.tolist()) if keep],
            "counts": counts[present].numpy(),
            "edge_label": "mean signed edge weight w" if continuous else "P(edge positive)",
            "edge_low": -1.0 if continuous else 0.0,
            "sign": means[0].numpy(), "positive": means[1].numpy(),
            "negative": means[2].numpy(), "example_sign": example[0].numpy(),
            "example_positive": example[1].numpy(), "example_negative": example[2].numpy()}


@torch.no_grad()
def collect_graphs(cfg: Config, model: nn.Module,
                   loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The learned graphs of every window in a loader.

    Parameters
    ----------
    cfg : Config
        Run settings (device).
    model : nn.Module
        Trained network.
    loader : DataLoader
        Windows to run.

    Returns
    -------
    stable : np.ndarray
        Stable graph ``[C, C]``.
    deltas : np.ndarray
        Change of every window ``[N, T, C, C]``.
    stages : np.ndarray
        Stage of every window ``[N]``.
    """
    model.eval()
    deltas, stages, stable = [], [], None
    for x, y, *extra in loader:
        extra = [item.to(cfg.device) for item in extra]
        _, delta, batch_stable = model(x.to(cfg.device), *extra)
        if stable is None:
            stable = batch_stable.cpu().numpy()
        deltas.append(delta.cpu().numpy())
        stages.append(y.numpy())
    return stable, np.concatenate(deltas), np.concatenate(stages)


def rolling_report(cfg: Config, model: nn.Module, loaders: dict[str, DataLoader]) -> dict:
    """Scores of a rolling-window run (Exp 13) beyond the per-second test scores.

    - ``per_epoch``: the predicted probabilities of the test windows whose
      last second lies in the same 30 s scored epoch are averaged, giving
      one prediction per epoch, comparable with work that scores epochs.
    - ``aligned``: only the windows that end on a multiple of
      ``data.train_stride_s`` and lie inside one scored epoch; for 5 s
      windows these are exactly Exp 12's test windows.
    - ``no_elapsed``: per-second scores with the elapsed time set to 0 (its
      training mean), and the drop; the model was trained with it, so the
      drop overstates its value.
    - ``elapsed_only``: a baseline that sees only the elapsed time: the
      stage frequencies of the training windows per whole hour since the
      recording started, divided by the overall stage frequencies (so it
      does not just predict N2), arg-maxed.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : nn.Module
        Trained network.
    loaders : dict
        Rolling loaders with ``train`` and ``test``.

    Returns
    -------
    dict
        The four entries above, each with the metrics of :func:`scores`
        (``no_elapsed`` also with ``delta_kappa`` and ``delta_macro_f1``).
    """
    test = loaders["test"].dataset
    true, probability = predict_proba(cfg, model, loaders["test"])
    report = {}

    epochs, position = np.unique(test.epoch_id, return_inverse=True)
    position = position.ravel()
    summed = np.zeros((len(epochs), probability.shape[1]))
    np.add.at(summed, position, probability)
    epoch_true = np.zeros(len(epochs), dtype=np.int64)
    epoch_true[position] = true
    report["per_epoch"] = {**scores(cfg, epoch_true, summed.argmax(-1)), "epochs": len(epochs)}

    aligned = test.aligned
    report["aligned"] = {**scores(cfg, true[aligned], probability[aligned].argmax(-1)),
                         "windows": int(aligned.sum())}

    base = scores(cfg, true, probability.argmax(-1))
    _, blank = predict_proba(cfg, model, loaders["test"], zero_elapsed=True)
    without = scores(cfg, true, blank.argmax(-1))
    report["no_elapsed"] = {**without, "delta_kappa": without["kappa"] - base["kappa"],
                            "delta_macro_f1": without["macro_f1"] - base["macro_f1"]}

    train = loaders["train"].dataset
    labelled = train.all_stage >= 0
    hours, stages = np.floor(train.all_elapsed[labelled]).astype(int), train.all_stage[labelled]
    n_stages = len(cfg.data.stages)
    overall = np.bincount(stages, minlength=n_stages) / len(stages)
    table = np.zeros((hours.max() + 1, n_stages))
    np.add.at(table, (hours, stages), 1.0)
    choice = (table / np.maximum(overall, 1e-12)).argmax(-1)
    test_hours = np.minimum(np.floor(test.elapsed).astype(int), len(choice) - 1)
    report["elapsed_only"] = scores(cfg, true, choice[test_hours])
    return report

