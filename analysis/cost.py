"""Size, compute, memory and latency of GraphChangeNet.

``static`` measures what does not depend on other jobs sharing the machine: parameters per module,
floating-point operations per 30 s epoch (PyTorch ``FlopCounterMode``, which counts matrix products
and convolutions, 2 operations per multiply-add) and peak GPU memory of this process
(``torch.cuda.max_memory_allocated``). It does so for the trained Exp 15 at 83 electrodes and, on
synthetic input with random weights, for C = 19 … 256 electrodes on a unit sphere.

``timing`` measures wall-clock latency and throughput and must run on an idle GPU / CPU: one new
epoch at a time (the streaming cost: Levels 1 and 2 for the new epoch plus one step of the causal
context) and batched runs of 20 epochs as in evaluation.

argv: ``static|timing`` and the output JSON path.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet, load_weights  # noqa: E402

RUN = REPO / "runs/exp15_final_fold0"
ELECTRODES = (19, 32, 64, 83, 128, 192, 256)
MODULES = ("graph_learner", "graph_conv", "position", "sequence", "readout", "context", "head",
           "decoder")


def trained_model(device: torch.device) -> tuple[GraphChangeNet, object, int]:
    """Exp 15 (fold 0) with its trained weights, in evaluation mode.

    Parameters
    ----------
    device : torch.device
        Where to put the model.

    Returns
    -------
    model : GraphChangeNet
        The trained network.
    cfg : Config
        Its configuration.
    samples : int
        Samples per electrode in one 30 s epoch.
    """
    cfg = load_run_config(RUN, {})
    channels, coords = data.read_montage(data.find_montage(cfg.paths.data_root))
    model = GraphChangeNet(cfg, len(channels), coords).to(device)
    load_weights(model, RUN / "classify_best.pt", device)
    return model.eval(), cfg, cfg.data.n_segments * cfg.data.segment_len


def sphere(n_electrodes: int, seed: int) -> np.ndarray:
    """Random electrode positions on the upper half of a unit sphere.

    Parameters
    ----------
    n_electrodes : int
        Number of positions.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        ``[n_electrodes, 3]`` coordinates.
    """
    points = np.random.default_rng(seed).normal(size=(n_electrodes, 3))
    points[:, 2] = np.abs(points[:, 2])
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def parameters(model: GraphChangeNet) -> dict:
    """Parameter count of every slot of the network and of the staging path.

    Parameters
    ----------
    model : GraphChangeNet
        The network.

    Returns
    -------
    dict
        Slot name to parameters, plus ``total`` and ``staging`` (total without the stage-1 decoder).
    """
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in MODULES}
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["staging"] = counts["total"] - counts["decoder"]
    return counts


def flops(model: GraphChangeNet, x: torch.Tensor) -> tuple[int, dict]:
    """Operations of one forward pass, in total and per top-level slot.

    Parameters
    ----------
    model : GraphChangeNet
        The network.
    x : torch.Tensor
        Input runs ``[B, N, C, samples]``.

    Returns
    -------
    total : int
        Operations of the pass.
    per_module : dict
        Slot name to operations.
    """
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    counts = counter.get_flop_counts()
    per_module = {name: int(sum(counts.get(f"GraphChangeNet.{name}", {}).values()))
                  for name in MODULES}
    return int(counter.get_total_flops()), per_module


def peak_memory(model: GraphChangeNet, x: torch.Tensor) -> float:
    """Peak GPU memory of one forward pass of this process, in MiB, weights included.

    Parameters
    ----------
    model : GraphChangeNet
        The network, on the GPU.
    x : torch.Tensor
        Input runs on the GPU.

    Returns
    -------
    float
        Peak allocated memory in MiB.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def static(out: Path) -> None:
    """Parameters, operations and peak memory of Exp 15 and of the electrode-count sweep.

    Parameters
    ----------
    out : Path
        Output JSON path.
    """
    device = torch.device("cuda")
    model, cfg, samples = trained_model(device)
    n_channels = model.dims.n_channels
    result = {"run": str(RUN.relative_to(REPO)), "gpu": torch.cuda.get_device_name(0),
              "samples_per_epoch": samples, "parameters": parameters(model)}
    stream = torch.randn(1, 1, n_channels, samples, device=device)
    batch = torch.randn(cfg.finetune.batch_size, cfg.data.context_epochs, n_channels, samples,
                        device=device)
    total, per_module = flops(model, stream)
    batch_total, _ = flops(model, batch)
    result["per_epoch"] = {
        "gflops_streaming": total / 1e9,
        "gflops_per_module": {k: v / 1e9 for k, v in per_module.items()},
        "gflops_per_epoch_in_evaluation_batch": batch_total / batch.shape[0] / batch.shape[1] / 1e9,
        "peak_mib_streaming": peak_memory(model, stream),
        "peak_mib_evaluation_batch": peak_memory(model, batch),
        "evaluation_batch": list(batch.shape),
        "weights_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
        "input_mib_per_epoch": n_channels * samples * 4 / 2**20}
    del model
    sweep = []
    for n_electrodes in ELECTRODES:
        torch.manual_seed(0)
        synthetic = GraphChangeNet(cfg, n_electrodes, sphere(n_electrodes, 0)).to(device).eval()
        x = torch.randn(1, 1, n_electrodes, samples, device=device)
        total, _ = flops(synthetic, x)
        sweep.append({"electrodes": n_electrodes,
                      "parameters": sum(p.numel() for p in synthetic.parameters()),
                      "gflops_per_epoch": total / 1e9,
                      "peak_mib_streaming": peak_memory(synthetic, x)})
        del synthetic
    result["electrode_sweep"] = sweep
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def clock(model: GraphChangeNet, x: torch.Tensor, device: torch.device, repeats: int) -> dict:
    """Median and spread of the forward time over repeated passes after a warm-up.

    Parameters
    ----------
    model : GraphChangeNet
        The network.
    x : torch.Tensor
        Input runs.
    device : torch.device
        Device of the model.
    repeats : int
        Timed passes.

    Returns
    -------
    dict
        ``median_ms``, ``p10_ms``, ``p90_ms`` per pass.
    """
    times = []
    with torch.no_grad():
        for index in range(repeats + 5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if index >= 5:
                times.append((time.perf_counter() - start) * 1000)
    q10, q50, q90 = np.percentile(times, [10, 50, 90])
    return {"median_ms": float(q50), "p10_ms": float(q10), "p90_ms": float(q90)}


def timing(out: Path) -> None:
    """Latency per epoch (streaming, one epoch at a time) and batched throughput, GPU and CPU.

    Parameters
    ----------
    out : Path
        Output JSON path.
    """
    result = {}
    for name in ("cuda", "cpu"):
        device = torch.device(name)
        model, cfg, samples = trained_model(device)
        n_channels = model.dims.n_channels
        stream = torch.randn(1, 1, n_channels, samples, device=device)
        batch = torch.randn(cfg.finetune.batch_size, cfg.data.context_epochs, n_channels, samples,
                            device=device)
        repeats = 50 if name == "cuda" else 10
        streaming = clock(model, stream, device, repeats)
        batched = clock(model, batch, device, max(3, repeats // 5))
        epochs = batch.shape[0] * batch.shape[1]
        result[name] = {"device": torch.cuda.get_device_name(0) if name == "cuda" else "cpu",
                        "threads": torch.get_num_threads(),
                        "streaming_ms_per_epoch": streaming,
                        "batched_ms_per_batch": batched, "batch": list(batch.shape),
                        "batched_epochs_per_second": epochs / batched["median_ms"] * 1000,
                        "night_of_900_epochs_seconds": 900 * streaming["median_ms"] / 1000}
        del model
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"static": static, "timing": timing}[sys.argv[1]](Path(sys.argv[2]))
