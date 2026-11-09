from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from evaluate import curve_scores, evaluate


def mask_neighbours(cfg: Config, model: nn.Module) -> torch.Tensor | None:
    """The electrodes hidden together in ``pretrain.mask_mode: cluster``.

    Parameters
    ----------
    cfg : Config
        Run settings (``pretrain.mask_mode``, ``pretrain.cluster_size``).
    model : nn.Module
        Network whose ``dims.coordinates`` give the electrode positions.

    Returns
    -------
    torch.Tensor or None
        ``[C, cluster_size + 1]`` indices, each electrode first, then its
        nearest neighbours by scalp distance; None in ``random`` mode.

    Raises
    ------
    ValueError
        If ``pretrain.mask_mode`` is unknown.
    """
    if cfg.pretrain.mask_mode == "random":
        return None
    if cfg.pretrain.mask_mode != "cluster":
        raise ValueError(f"pretrain.mask_mode must be random or cluster, not {cfg.pretrain.mask_mode!r}")
    coordinates = torch.tensor(model.dims.coordinates, device=cfg.device)
    order = torch.cdist(coordinates, coordinates).argsort(dim=1)
    return order[:, :cfg.pretrain.cluster_size + 1]


def sample_mask(cfg: Config, batch: int, n_channels: int, device: torch.device,
                channel_prob: float | None = None,
                neighbours: torch.Tensor | None = None) -> torch.Tensor:
    """Draw a hiding mask that mixes random cells with whole electrodes.

    Random cells can be filled in from the same electrode's neighbouring
    samples; hiding a whole electrode for the entire window forces the answer
    to come through the graph.

    Parameters
    ----------
    cfg : Config
        Run settings (segments, mask ratio, whole-electrode share).
    batch : int
        Windows in the batch.
    n_channels : int
        Electrodes.
    device : torch.device
        Device of the mask.
    channel_prob : float, optional
        Share of windows with whole-electrode masking; defaults to
        ``cfg.pretrain.channel_mask_prob``.
    neighbours : torch.Tensor, optional
        From :func:`mask_neighbours`: whole-electrode masks then hide clusters
        of an electrode and its nearest neighbours. Each electrode is a centre
        with probability ``1 - (1 - mask_ratio) ** (1 / cluster)``, so that,
        with clusters overlapping, about ``mask_ratio`` of the electrodes are
        hidden, as in random mode.

    Returns
    -------
    torch.Tensor
        Boolean mask ``[batch, n_segments, n_channels]``, true where hidden.
    """
    if channel_prob is None:
        channel_prob = cfg.pretrain.channel_mask_prob
    ratio, n_segments = cfg.pretrain.mask_ratio, cfg.data.n_segments
    cells = torch.rand(batch, n_segments, n_channels, device=device) < ratio
    if neighbours is None:
        whole = (torch.rand(batch, 1, n_channels, device=device) < ratio)
    else:
        centres = torch.rand(batch, n_channels, device=device) < 1 - (1 - ratio) ** (1 / neighbours.shape[1])
        whole = torch.zeros(batch, n_channels, dtype=torch.bool, device=device)
        rows, cols = centres.nonzero(as_tuple=True)
        whole[rows.unsqueeze(1), neighbours[cols]] = True
        whole = whole.unsqueeze(1)
    whole = whole.expand(batch, n_segments, n_channels)
    pick = torch.rand(batch, 1, 1, device=device) < channel_prob
    return torch.where(pick, whole, cells)


def reconstruction_loss(cfg: Config, rebuilt: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Masked smooth-L1 reconstruction loss, absolute or relative to predicting zeros.

    With ``pretrain.relative_losses`` the loss is divided by the
    smooth-L1 loss of predicting zeros on the same hidden cells, detached.
    The ratio is 1 for the trivial prediction and 0 for a perfect one, does
    not depend on the subject's amplitude, and compares training and
    validation subjects on one scale.

    Parameters
    ----------
    cfg : Config
        Run settings (``pretrain.relative_losses``).
    rebuilt, target : torch.Tensor
        Reconstructed and original segments ``[B, T, C, L]``.
    mask : torch.Tensor
        ``[B, T, C]``, true where hidden.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    loss = F.smooth_l1_loss(rebuilt[mask], target[mask])
    if not cfg.pretrain.relative_losses:
        return loss
    baseline = F.smooth_l1_loss(torch.zeros_like(target[mask]), target[mask]).detach()
    return loss / baseline.clamp(min=1e-8)


@torch.no_grad()
def val_masked_loss(cfg: Config, model: nn.Module, loader: DataLoader, seed: int = 0) -> float:
    """Masked reconstruction loss on held-out windows under a fixed mask.

    The global random state is saved and restored, so this call does not
    change the training stream.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : nn.Module
        Network being pretrained.
    loader : DataLoader
        Validation windows.
    seed : int
        Seed of the mask, identical at every call so epochs are comparable.

    Returns
    -------
    float
        Mean :func:`reconstruction_loss` over the batches.
    """
    device = cfg.device
    was_training = model.training
    model.eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    total = 0.0
    for x, *_ in loader:
        x = x.to(device)
        mask = sample_mask(cfg, x.shape[0], x.shape[1], device,
                           neighbours=mask_neighbours(cfg, model))
        rebuilt, _ = model.reconstruct(x, mask)
        total += reconstruction_loss(cfg, rebuilt, model.segments(x), mask).item()

    torch.random.set_rng_state(state)
    model.train(was_training)
    return total / len(loader)


@torch.no_grad()
def baselines(cfg: Config, model: nn.Module, loader: DataLoader, n_batches: int = 10,
              seed: int = 0) -> dict[str, float]:
    """Masked loss of the model next to predictors that use no learned graph.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : nn.Module
        Network being pretrained.
    loader : DataLoader
        Windows to score.
    n_batches : int
        Batches to average over.
    seed : int
        Seed of the mask; the global random state is restored afterwards.

    Returns
    -------
    dict
        Mean loss of ``model``, ``zeros`` and ``visible_mean``.
    """
    device = cfg.device
    was_training = model.training
    model.eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    out: dict[str, list[float]] = {"model": [], "zeros": [], "visible_mean": []}

    for index, (x, *_) in enumerate(loader):
        x = x.to(device)
        target = model.segments(x)
        mask = sample_mask(cfg, x.shape[0], x.shape[1], device,
                           neighbours=mask_neighbours(cfg, model))

        rebuilt, _ = model.reconstruct(x, mask)
        seen = (~mask).unsqueeze(-1)
        visible = (target * seen).sum(2) / seen.sum(2).clamp(min=1)
        visible = visible.unsqueeze(2).expand_as(target)

        out["model"].append(F.smooth_l1_loss(rebuilt[mask], target[mask]).item())
        out["zeros"].append(F.smooth_l1_loss(torch.zeros_like(target[mask]), target[mask]).item())
        out["visible_mean"].append(F.smooth_l1_loss(visible[mask], target[mask]).item())
        if index + 1 >= n_batches:
            break

    torch.random.set_rng_state(state)
    model.train(was_training)
    return {name: round(float(np.mean(values)), 4) for name, values in out.items()}


def band_magnitude(series: torch.Tensor, size: int, band: torch.Tensor) -> torch.Tensor:
    """STFT magnitude inside a frequency band, with a floor that keeps its gradient finite.

    Parameters
    ----------
    series : torch.Tensor
        Signals ``[N, samples]``.
    size : int
        FFT size; Hann window of that size, hop of a quarter of it.
    band : torch.Tensor
        Boolean over the ``size // 2 + 1`` frequency bins.

    Returns
    -------
    torch.Tensor
        ``[N, bins in band, frames]``.
    """
    window = torch.hann_window(size, device=series.device)
    spectrum = torch.stft(series, size, hop_length=size // 4, window=window, return_complex=True)
    return torch.view_as_real(spectrum).pow(2).sum(-1).add(1e-10).sqrt()[:, band]


def spectral_loss(cfg: Config, rebuilt: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss on the electrodes that have hidden cells.

    The waveform loss punishes every phase error, so the decoder learns to
    smooth; comparing magnitude spectra rewards getting the rhythms right
    (Yamamoto et al., ICASSP 2020). Each electrode's signal is rebuilt from
    the original with its hidden cells replaced by the reconstruction, so
    only the hidden cells differ. For every FFT size, spectral convergence
    plus the mean absolute difference of log magnitudes, over the pass band
    ``data.highpass`` to ``data.lowpass``. The logarithms get a floor of
    1e-5 plus ``pretrain.spectral_floor`` times the electrode's mean target
    magnitude; with 0 the floor is the absolute 1e-5 of Exp 13-14.

    Parameters
    ----------
    cfg : Config
        Run settings (``pretrain.stft_sizes``, ``data``).
    rebuilt, target : torch.Tensor
        Reconstructed and original segments ``[B, T, C, L]``.
    mask : torch.Tensor
        ``[B, T, C]``, true where hidden.

    Returns
    -------
    torch.Tensor
        Scalar loss, averaged over the FFT sizes.
    """
    rows = mask.any(1)
    if not rows.any():
        return rebuilt.sum() * 0.0
    signal = target.permute(0, 2, 1, 3).flatten(2)[rows]
    estimate = torch.where(mask.unsqueeze(-1), rebuilt, target).permute(0, 2, 1, 3).flatten(2)[rows]
    data, total = cfg.data, 0.0
    for size in cfg.pretrain.stft_sizes:
        frequency = torch.fft.rfftfreq(size, 1.0 / data.sfreq).to(signal.device)
        band = (frequency >= data.highpass) & (frequency <= data.lowpass)
        true = band_magnitude(signal, size, band)
        guess = band_magnitude(estimate, size, band)
        convergence = torch.linalg.norm(true - guess) / torch.linalg.norm(true).clamp(min=1e-8)
        floor = 1e-5 + cfg.pretrain.spectral_floor * true.mean((1, 2), keepdim=True).detach()
        log_distance = (torch.log(true + floor) - torch.log(guess + floor)).abs().mean()
        total = total + convergence + log_distance
    return total / len(cfg.pretrain.stft_sizes)


def future_loss(cfg: Config, predictors: nn.ModuleList, context: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
    """Contrastive prediction of later segments from the causal outputs (CPC).

    The output of segment ``t`` (which saw segments ``1..t`` only) predicts,
    through one linear map per step ``k``, the token of segment ``t + k``.
    InfoNCE (van den Oord et al., 2018): cosine similarity over a temperature
    against every token of every window in the batch, the true one being the
    target.

    Parameters
    ----------
    cfg : Config
        Run settings (``pretrain.future_temperature``).
    predictors : nn.ModuleList
        One ``Linear(H, H)`` per step.
    context : torch.Tensor
        Sequence outputs ``[B, T, H]``.
    tokens : torch.Tensor
        Segment tokens ``[B, T, H]``.

    Returns
    -------
    torch.Tensor
        Scalar loss, averaged over the steps.
    """
    batch, n_segments, _ = context.shape
    targets = F.normalize(tokens.flatten(0, 1), dim=-1)
    total, steps = 0.0, 0
    for step, predictor in enumerate(predictors, start=1):
        if step >= n_segments:
            break
        predicted = F.normalize(predictor(context[:, :n_segments - step]), dim=-1).flatten(0, 1)
        logits = predicted @ targets.T / cfg.pretrain.future_temperature
        window = torch.arange(batch, device=context.device).repeat_interleave(n_segments - step)
        position = torch.arange(n_segments - step, device=context.device).repeat(batch)
        total = total + F.cross_entropy(logits, window * n_segments + position + step)
        steps += 1
    return total / max(steps, 1)


class PretrainObjectives(nn.Module):
    """The stage-1 losses of Exp 13 and their weighting.

    Masked reconstruction (:func:`reconstruction_loss`), plus a spectral
    loss (:func:`spectral_loss`) and a "predict the future" loss
    (:func:`future_loss`) when enabled. Each loss is first multiplied by its
    fixed weight (``pretrain.<task>_weight``). With ``uncertainty_weighting``
    every weighted loss ``L_i`` then enters as ``0.5 exp(-s_i) L_i + 0.5 s_i``
    with a learned ``s_i = log sigma_i^2`` starting at 0 (Kendall et al.,
    CVPR 2018, as used by MTSSRL-MD); otherwise the weighted losses are
    summed. The optimum of that weighting is ``exp(s_i) = L_i``, so a loss
    that approaches 0 gets an ever larger weight; Exp 15 uses fixed weights.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.pretrain``).
    width : int
        Width of the sequence outputs and tokens.
    """

    def __init__(self, cfg: Config, width: int):
        super().__init__()
        stage = cfg.pretrain
        self.cfg = cfg
        self.names = ["reconstruction"]
        if stage.spectral_loss:
            self.names.append("spectral")
        if stage.future_steps:
            self.names.append("future")
        self.predictors = nn.ModuleList(nn.Linear(width, width) for _ in range(stage.future_steps))
        self.log_variance = (nn.Parameter(torch.zeros(len(self.names)))
                             if stage.uncertainty_weighting else None)
        self.fixed = {"reconstruction": stage.reconstruction_weight,
                      "spectral": stage.spectral_weight, "future": stage.future_weight}
        self.last: dict[str, float] = {}

    def forward(self, model: nn.Module, x: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Every enabled stage-1 loss from one masked forward pass, weighted.

        Parameters
        ----------
        model : nn.Module
            Network being pretrained; the future loss needs a sequence
            component with an ``embed`` layer.
        x : torch.Tensor
            Windows ``[B, C, samples]``.
        mask : torch.Tensor
            ``[B, T, C]``, true where hidden.

        Returns
        -------
        loss : torch.Tensor
            Weighted total.
        delta : torch.Tensor
            The change ``[B, T, C, C]``, for its penalty.
        """
        losses, delta = self.raw_losses(model, x, mask)
        self.last = {name: float(value.detach()) for name, value in zip(self.names, losses)}
        losses = [self.fixed[name] * value for name, value in zip(self.names, losses)]
        if self.log_variance is None:
            return sum(losses), delta
        weighted = [0.5 * torch.exp(-s) * value + 0.5 * s
                    for s, value in zip(self.log_variance, losses)]
        return sum(weighted), delta

    def raw_losses(self, model: nn.Module, x: torch.Tensor,
                   mask: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Every enabled stage-1 loss, unweighted, from one masked forward pass.

        Parameters
        ----------
        model : nn.Module
            Network being pretrained.
        x : torch.Tensor
            Windows ``[B, C, samples]``.
        mask : torch.Tensor
            ``[B, T, C]``, true where hidden.

        Returns
        -------
        losses : list of torch.Tensor
            One scalar per name in ``names``, relative to its trivial baseline
        with ``pretrain.relative_losses``.
        delta : torch.Tensor
            The change ``[B, T, C, C]``.
        """
        rebuilt, delta, hidden, context = model.pretrain_outputs(x, mask)
        target = model.segments(x)
        relative = self.cfg.pretrain.relative_losses
        losses = [reconstruction_loss(self.cfg, rebuilt, target, mask)]
        if "spectral" in self.names:
            value = spectral_loss(self.cfg, rebuilt, target, mask)
            if relative:
                with torch.no_grad():
                    trivial = spectral_loss(self.cfg, torch.zeros_like(rebuilt), target, mask)
                value = value / trivial.clamp(min=1e-8)
            losses.append(value)
        if "future" in self.names:
            tokens = model.sequence.embed(hidden.flatten(2))
            value = future_loss(self.cfg, self.predictors, context, tokens)
            if relative:
                value = value / float(np.log(tokens.shape[0] * tokens.shape[1]))
            losses.append(value)
        return losses, delta

    def summary(self) -> dict:
        """Last raw losses and, with uncertainty weighting, the learned task weights.

        Returns
        -------
        dict
            ``losses`` per task and ``weights`` (``0.5 exp(-s_i)``) per task.
        """
        weights = ({} if self.log_variance is None else
                   {name: float(0.5 * torch.exp(-s)) for name, s in
                    zip(self.names, self.log_variance.detach().cpu())})
        return {"losses": self.last, "weights": weights}


@torch.no_grad()
def stage1_curves(cfg: Config, model: nn.Module, objectives: "PretrainObjectives | None",
                  loader: DataLoader, seed: int = 0) -> dict[str, float]:
    """Every stage-1 loss on a fixed set of batches, each relative to a trivial baseline.

    Scored like the validation loss (eval mode, no augmentation, the same
    masks at every call), so a training subset and the validation set can be
    drawn on one axis. Each loss is divided by what a trivial prediction
    scores on the same batch, so 1 means "no better than trivial" whatever
    the subject's amplitude: reconstruction by predicting zeros, the
    spectral loss by the spectrum of the signal with its hidden cells set to
    zero, and the contrastive loss by ``log`` of its number of candidates
    (chance). The absolute reconstruction loss is kept as well. The random
    state of every device is restored afterwards.

    Parameters
    ----------
    cfg : Config
        Run settings (``evaluation.curve_batches``).
    model : nn.Module
        Network being pretrained.
    objectives : PretrainObjectives or None
        The stage-1 losses; None scores the reconstruction alone.
    loader : DataLoader
        Epochs to score, in a fixed order.
    seed : int
        Seed of the masks.

    Returns
    -------
    dict
        Mean relative loss per task, and ``reconstruction_absolute``.
    """
    device = cfg.device
    was_training = model.training
    model.eval()
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    names = objectives.names if objectives is not None else ["reconstruction"]
    sums = {name: 0.0 for name in names + ["reconstruction_absolute"]}
    count = 0
    for index, (x, *_) in enumerate(loader):
        if index >= cfg.evaluation.curve_batches:
            break
        x = x.to(device)
        mask = sample_mask(cfg, x.shape[0], x.shape[1], device,
                           neighbours=mask_neighbours(cfg, model))
        rebuilt, _, hidden, context = model.pretrain_outputs(x, mask)
        target = model.segments(x)
        absolute = F.smooth_l1_loss(rebuilt[mask], target[mask])
        zeros = F.smooth_l1_loss(torch.zeros_like(target[mask]), target[mask])
        sums["reconstruction_absolute"] += absolute.item()
        sums["reconstruction"] += (absolute / zeros.clamp(min=1e-8)).item()
        if "spectral" in names:
            trivial = spectral_loss(cfg, torch.zeros_like(rebuilt), target, mask)
            sums["spectral"] += (spectral_loss(cfg, rebuilt, target, mask)
                                 / trivial.clamp(min=1e-8)).item()
        if "future" in names:
            tokens = model.sequence.embed(hidden.flatten(2))
            chance = float(np.log(tokens.shape[0] * tokens.shape[1]))
            value = future_loss(cfg, objectives.predictors, context, tokens).item()
            sums["future"] += value / chance
        count += 1

    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    model.train(was_training)
    return {name: value / max(count, 1) for name, value in sums.items()}


def pretrain(cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             checkpoint: Path, train_eval: DataLoader | None) -> list:
    """Stage 1: train the network to restore hidden parts of the signal.

    Cosine learning-rate decay; the best weights by validation masked loss
    are saved to ``checkpoint`` and loaded back at the end. With the Exp 13
    settings (``spectral_loss``, ``future_steps``, ``uncertainty_weighting``)
    the training loss is that of :class:`PretrainObjectives`, whose final
    task weights go to ``pretrain_objectives.json`` next to the checkpoint;
    model selection still uses the masked reconstruction loss alone. Then
    the logged training loss is the reconstruction part, comparable with the
    validation loss (the weighted total is shown as ``total``), the gradient
    is clipped over the loss weights and predictors too, and a step whose
    loss or gradient is not finite is skipped and counted (``skipped``)
    instead of writing NaN into the weights; with reconstruction alone the
    same check runs when ``pretrain.max_skip_share`` is below 1, so the
    single-task stage 1 of Exp 15c is guarded like Exp 15. The logged
    training loss is the mean over the steps that were taken, and an epoch
    whose share of skipped steps exceeds ``pretrain.max_skip_share`` stops
    the run with an error, because the rest of it would train on a biased
    subset of the batches.

    With ``evaluation.curve_batches`` above 0, every validation pass also
    scores every stage-1 loss with :func:`stage1_curves` on the training
    subset ``train_eval`` and on the validation epochs, and the history rows
    become dictionaries with those curves and the epoch's skipped steps.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.pretrain``).
    model : nn.Module
        Network, trained in place.
    train_loader, val_loader : DataLoader
        Training and validation windows.
    checkpoint : Path
        Where the best weights are saved.
    train_eval : DataLoader or None
        Training epochs without augmentation in a fixed order, for the
        curves; None without curves.

    Returns
    -------
    list
        ``(epoch, train_loss, val_loss)`` at every validation pass, or a
        dictionary per pass with the curves.

    Raises
    ------
    RuntimeError
        If an epoch skips more than ``pretrain.max_skip_share`` of its steps.
    """
    stage, device = cfg.pretrain, cfg.device
    objectives = None
    parameters = list(model.parameters())
    if stage.spectral_loss or stage.future_steps or stage.uncertainty_weighting:
        objectives = PretrainObjectives(cfg, model.sequence.output_dim).to(device)
        parameters += list(objectives.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=stage.lr, weight_decay=stage.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=stage.epochs)

    history, best, stale = [], float("inf"), 0
    epoch_bar = tqdm(range(1, stage.epochs + 1), desc="pretrain")
    skipped = 0
    curves = cfg.evaluation.curve_batches > 0 and train_eval is not None
    for epoch in epoch_bar:
        model.train()
        running, total, taken, skipped_epoch = 0.0, 0.0, 0, 0
        for x, *_ in tqdm(train_loader, leave=False, desc=f"epoch {epoch}"):
            x = x.to(device)
            mask = sample_mask(cfg, x.shape[0], x.shape[1], device,
                           neighbours=mask_neighbours(cfg, model))
            if objectives is None:
                rebuilt, delta = model.reconstruct(x, mask)
                loss = reconstruction_loss(cfg, rebuilt, model.segments(x), mask)
            else:
                loss, delta = objectives(model, x, mask)
            loss = loss + stage.delta_penalty * delta.abs().mean()
            if stage.graph_penalty:
                loss = loss + stage.graph_penalty * model.penalty()

            optimiser.zero_grad()
            loss.backward()
            if objectives is None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                guarded = stage.max_skip_share < 1.0
                if guarded and not (torch.isfinite(loss) and torch.isfinite(norm)):
                    optimiser.zero_grad()
                    skipped += 1
                    skipped_epoch += 1
                    continue
                optimiser.step()
                running += loss.item()
                taken += 1
                continue
            norm = nn.utils.clip_grad_norm_(parameters, 1.0)
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                optimiser.zero_grad()
                skipped += 1
                skipped_epoch += 1
                continue
            optimiser.step()
            running += objectives.last["reconstruction"]
            total += loss.item()
            taken += 1
        scheduler.step()

        share = skipped_epoch / max(len(train_loader), 1)
        if share > stage.max_skip_share:
            raise RuntimeError(f"stage 1 epoch {epoch}: {skipped_epoch} of {len(train_loader)} "
                               f"steps were not finite ({share:.1%}, the limit is "
                               f"{stage.max_skip_share:.1%})")
        train_loss = running / max(taken, 1)
        shown = {"train": f"{train_loss:.4f}"}
        if objectives is not None:
            shown.update(total=f"{total / max(taken, 1):.3f}", skipped=skipped)
        if epoch % stage.val_every == 0 or epoch == stage.epochs:
            val_loss = val_masked_loss(cfg, model, val_loader)
            if curves:
                history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                                "skipped": skipped_epoch, "steps": len(train_loader),
                                "train_curves": stage1_curves(cfg, model, objectives, train_eval),
                                "val_curves": stage1_curves(cfg, model, objectives, val_loader)})
            else:
                history.append((epoch, train_loss, val_loss))
            if val_loss < best - 1e-4:
                best, stale = val_loss, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += stage.val_every
            epoch_bar.set_postfix(**shown, val=f"{val_loss:.4f}", best=f"{best:.4f}", stale=stale)
            if stale >= stage.patience:
                epoch_bar.write(f"stopped at epoch {epoch}: no gain for {stale} epochs")
                break
        else:
            epoch_bar.set_postfix(**shown, best=f"{best:.4f}", stale=stale)

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    if objectives is not None:
        checkpoint.with_name("pretrain_objectives.json").write_text(
            json.dumps({**objectives.summary(), "skipped_steps": skipped}, indent=2))
    return history


def build_optimiser(cfg: Config, model: nn.Module, frozen: bool) -> torch.optim.Optimizer:
    """Optimise the head alone, or everything with a slower backbone.

    The readout and the context component count as part of the head: they
    turn the sequence outputs into the head's input and are not trained in
    stage 1.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.finetune``).
    model : nn.Module
        Network being trained.
    frozen : bool
        Whether the backbone stays frozen.

    Returns
    -------
    torch.optim.Optimizer
        AdamW over the parameters that should move.
    """
    stage = cfg.finetune
    trained = (list(model.head.parameters()) + list(model.readout.parameters())
               + list(model.context.parameters()))
    head = set(map(id, trained))
    backbone = [p for p in model.parameters() if id(p) not in head]
    for p in backbone:
        p.requires_grad_(not frozen)

    groups = [{"params": trained, "lr": stage.lr}]
    if not frozen:
        groups.append({"params": backbone, "lr": stage.lr * stage.backbone_lr_scale})
    return torch.optim.AdamW(groups, lr=stage.lr, weight_decay=stage.weight_decay)


def build_scheduler(cfg: Config, optimiser: torch.optim.Optimizer, epochs: int):
    """Linear warm-up followed by a cosine decay.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.finetune.warmup_epochs``).
    optimiser : torch.optim.Optimizer
        Optimiser to schedule.
    epochs : int
        Epochs the schedule spans.

    Returns
    -------
    SequentialLR
        The chained scheduler.
    """
    warmup = cfg.finetune.warmup_epochs
    return SequentialLR(optimiser, [
        LinearLR(optimiser, start_factor=0.1, total_iters=warmup),
        CosineAnnealingLR(optimiser, T_max=max(epochs - warmup, 1))],
        milestones=[warmup])


def class_balanced_weights(cfg: Config, counts: np.ndarray) -> torch.Tensor:
    """Bounded effective-number class weights (Cui et al., 2019).

    Parameters
    ----------
    cfg : Config
        Run settings (beta and bounds in ``cfg.finetune``).
    counts : np.ndarray
        Training windows per stage.

    Returns
    -------
    torch.Tensor
        One weight per stage, mean-normalised and clamped.
    """
    stage = cfg.finetune
    counts = torch.as_tensor(counts, dtype=torch.float32).clamp_min(1)
    beta = torch.tensor(stage.class_balance_beta, dtype=torch.float32)
    weights = (1.0 - beta) / (1.0 - beta.pow(counts))
    weights = weights / weights.mean()
    return weights.clamp(stage.class_weight_min, stage.class_weight_max)


def classification_loss(cfg: Config, logits: torch.Tensor, target: torch.Tensor,
                        class_weights: torch.Tensor | None) -> torch.Tensor:
    """Label-smoothed, class-weighted focal cross entropy.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.finetune``).
    logits : torch.Tensor
        ``[B, n_classes]``.
    target : torch.Tensor
        Stage indices ``[B]``.
    class_weights : torch.Tensor, optional
        One weight per stage.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.
    """
    stage = cfg.finetune
    loss = F.cross_entropy(logits, target, weight=class_weights,
                           label_smoothing=stage.label_smoothing, reduction="none")
    if stage.focal_gamma > 0:
        probability = logits.softmax(-1).gather(1, target.unsqueeze(1)).squeeze(1)
        loss = loss * (1.0 - probability).pow(stage.focal_gamma)
    return loss.mean()


def classify(cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             checkpoint: Path, train_eval: DataLoader | None) -> tuple[list[dict], float]:
    """Stage 2: train the head alone, then the whole network.

    The head is the only part stage 1 did not train, so the frozen phase is a
    linear probe of what pretraining produced. Model selection uses an
    exponential moving average of the weights and validation macro-F1.
    Runs of epochs (``data.context_epochs`` above 1) give one logit row per
    epoch; positions labelled -1 (lights on, artifacts) are left out of the
    loss, and with single windows every position counts, as before. A batch
    with no labelled position still steps (its loss is NaN, its gradient is
    the change penalty's alone), but is left out of the logged training loss.

    With ``evaluation.curve_batches`` above 0, every epoch scores the
    averaged weights with :func:`evaluate.curve_scores` on the whole
    validation set (``val_curves``, whose scores also select the model, as
    :func:`evaluate.evaluate` would) and on that many batches of
    ``train_eval`` (``train_curves``), so training and validation are
    compared with the same function.

    Parameters
    ----------
    cfg : Config
        Run settings (``cfg.finetune``).
    model : nn.Module
        Network, trained in place; ends with the best averaged weights.
    train_loader, val_loader : DataLoader
        Training and validation windows.
    checkpoint : Path
        Where the best averaged weights are saved.
    train_eval : DataLoader or None
        Training epochs without augmentation in a fixed order, for the
        curves; None without curves.

    Returns
    -------
    history : list of dict
        Validation metrics and training loss per epoch.
    best : float
        Best validation macro-F1.
    """
    stage, device = cfg.finetune, cfg.device
    averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(stage.ema_decay))
    optimiser = build_optimiser(cfg, model, frozen=stage.freeze_epochs > 0)
    scheduler = build_scheduler(cfg, optimiser, stage.epochs)
    train_counts = np.bincount(train_loader.dataset.y, minlength=len(cfg.data.stages))
    class_weights = (class_balanced_weights(cfg, train_counts).to(device)
                     if stage.class_balanced_loss else None)

    best, stale, history = -1.0, 0, []
    bar = tqdm(range(stage.epochs), desc="classify")
    for epoch in bar:
        if epoch == stage.freeze_epochs:
            optimiser = build_optimiser(cfg, model, frozen=False)
            scheduler = build_scheduler(cfg, optimiser, stage.epochs - epoch)
            bar.write(f"epoch {epoch}: backbone unfrozen")

        model.train()
        running, counted = 0.0, 0
        for x, y, *extra in tqdm(train_loader, leave=False, desc=f"epoch {epoch + 1}"):
            logits, delta, _ = model(x.to(device), *[item.to(device) for item in extra])
            target = y.to(device).flatten()
            logits = logits.flatten(0, -2)
            labelled = target >= 0
            loss = classification_loss(cfg, logits[labelled], target[labelled], class_weights)
            loss = loss + stage.delta_penalty * delta.abs().mean()
            if stage.graph_penalty:
                loss = loss + stage.graph_penalty * model.penalty()

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            averaged.update_parameters(model)
            if labelled.any():
                running += loss.item()
                counted += 1
        scheduler.step()

        if cfg.evaluation.curve_batches > 0 and train_eval is not None:
            val_curves = curve_scores(cfg, averaged.module, val_loader, 0)
            metrics = {key: value for key, value in val_curves.items() if key != "cross_entropy"}
            metrics["val_curves"] = val_curves
            metrics["train_curves"] = curve_scores(cfg, averaged.module, train_eval,
                                                   cfg.evaluation.curve_batches)
        else:
            metrics = evaluate(cfg, averaged.module, val_loader)
        metrics["train_loss"] = running / max(counted, 1)
        history.append(metrics)
        if metrics["macro_f1"] > best + 1e-4:
            best, stale = metrics["macro_f1"], 0
            torch.save(averaged.module.state_dict(), checkpoint)
        else:
            stale += 1

        bar.set_postfix(phase="probe" if epoch < stage.freeze_epochs else "tune",
                        loss=f"{metrics['train_loss']:.3f}",
                        f1=f"{metrics['macro_f1']:.3f}", kappa=f"{metrics['kappa']:.3f}",
                        best=f"{best:.3f}", stale=stale)
        if stale >= stage.patience:
            bar.write(f"stopped at epoch {epoch + 1}: no gain for {stale} epochs")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return history, best
