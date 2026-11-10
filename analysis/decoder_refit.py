"""Decoder-only refit of the final (stage-2) GraphChangeNet, to test whether its graph and representations still support reconstruction.

Stage 2 fine-tunes the backbone for staging but never trains the stage-1 decoder, so the stage-1 decoder
no longer matches the fine-tuned features. This script freezes every weight of the final model
(``classify_best.pt`` of fold 0) except the decoder, trains the decoder on the training nights with the
stage-1 masks (``train.sample_mask``) and the stage-1 reconstruction and spectral losses of the run's
configuration (relative to their trivial predictors, with the configured weights; the next-second task has
no decoder part and is left out), keeps the pass with the lowest validation masked loss
(``train.val_masked_loss``, validation nights only) and saves the whole model. It checks that no weight
outside the decoder moved. ``stage1_stats.py`` then scores the saved weights on the test masks of Table V(c)
as ``Exp 15 after stage 2, decoder refit``. argv: output checkpoint path, number of passes.
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402
from train import mask_neighbours, reconstruction_loss, sample_mask, spectral_loss, val_masked_loss  # noqa: E402

RUN = REPO / "runs/exp15_final_fold0"


def stage1_loss(cfg, model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted reconstruction and spectral loss of one masked batch, as in stage 1.

    Parameters
    ----------
    cfg : Config
        Run settings.
    model : torch.nn.Module
        GraphChangeNet with the decoder being refitted.
    x : torch.Tensor
        Windows ``[B, C, samples]``.
    mask : torch.Tensor
        ``[B, T, C]``, true where hidden.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    rebuilt = model.pretrain_outputs(x, mask)[0]
    target = model.segments(x)
    loss = cfg.pretrain.reconstruction_weight * reconstruction_loss(cfg, rebuilt, target, mask)
    if cfg.pretrain.spectral_loss:
        value = spectral_loss(cfg, rebuilt, target, mask)
        with torch.no_grad():
            trivial = spectral_loss(cfg, torch.zeros_like(rebuilt), target, mask)
        loss = loss + cfg.pretrain.spectral_weight * value / trivial.clamp(min=1e-8)
    return loss


def main():
    """Refit the decoder, keep the best pass by validation loss and save the model."""
    out, passes = Path(sys.argv[1]), int(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(RUN, {})
    split = json.loads((RUN / "metrics.json").read_text())["split"]
    arrays, channels, coords = data.load_dataset(cfg)
    loaders = data.build_epoch_loaders(cfg, {s: arrays[s] for s in sum(split.values(), [])}, split)
    model = GraphChangeNet(cfg, len(channels), coords).to(cfg.device)
    load_weights(model, RUN / "classify_best.pt", cfg.device)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("decoder.")}
    decoder = [p for k, p in model.named_parameters() if k.startswith("decoder.")]
    for k, p in model.named_parameters():
        p.requires_grad_(k.startswith("decoder."))
    optimiser = torch.optim.AdamW(decoder, lr=cfg.pretrain.lr, weight_decay=cfg.pretrain.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=passes)
    model.eval()
    best = val_masked_loss(cfg, model, loaders["pretrain_val"])
    print(f"validation masked loss with the stage-1 decoder: {best:.4f}", flush=True)
    torch.save(model.state_dict(), out)
    history = [{"pass": 0, "val": best}]
    for step in range(1, passes + 1):
        running, count = 0.0, 0
        with torch.enable_grad():
            for x, *_ in loaders["pretrain"]:
                x = x.to(cfg.device)
                mask = sample_mask(cfg, x.shape[0], x.shape[1], cfg.device,
                                   neighbours=mask_neighbours(cfg, model))
                loss = stage1_loss(cfg, model, x, mask)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder, 1.0)
                optimiser.step()
                running, count = running + loss.item(), count + 1
        scheduler.step()
        val = val_masked_loss(cfg, model, loaders["pretrain_val"])
        history.append({"pass": step, "train": running / max(count, 1), "val": val})
        print(f"pass {step}: train {running / max(count, 1):.4f}, validation {val:.4f}", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), out)
    state = torch.load(out, map_location="cpu")
    moved = [k for k, v in frozen.items() if not torch.equal(v.cpu(), state[k].cpu())]
    if moved:
        raise RuntimeError(f"weights outside the decoder moved: {moved[:5]}")
    out.with_suffix(".json").write_text(json.dumps({"history": history, "best_val": best}, indent=1))
    print(f"saved {out} (best validation masked loss {best:.4f}); no weight outside the decoder moved")


if __name__ == "__main__":
    main()
