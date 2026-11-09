"""Size, compute, memory and latency of every neural baseline next to GraphChangeNet (the cost table).

Every baseline is rebuilt exactly as its runner builds it for the unified configuration (our 83
electrodes at 1 kHz, runs of 20 epochs ending at the scored epoch), by calling the runner with its
``ours.yaml`` and intercepting the network where the runner would start training; its trained
weights (``baselines/results/<model>/ours/classify_best.pt``) are then loaded. All models therefore
read the same tensor ``[B, 20, 83, 30000]``. GraphChangeNet (Exp 15, fold 0) is measured the same way.

Measured per model:

- parameters;
- operations to score one epoch as evaluated (one 20-epoch run per scored epoch, the protocol of
  every unified run), and per epoch amortised over the run (run operations / 20). PyTorch
  ``FlopCounterMode`` counts matrix products and convolutions at 2 operations per multiply-add;
  recurrent layers (``nn.GRU``, ``nn.LSTM``) and FFTs, which it does not count, are added
  analytically (recurrences from their sizes; an N-point complex FFT as 5 N log2 N operations, a
  real-input FFT as half of that);
- peak GPU memory of this process for one run (batch 1) and for the evaluation batch (2 runs);
- with ``timing``: latency of one run (batch 1) and throughput of the evaluation batch, GPU and CPU.
  Timing must run on an idle machine.

argv: ``static|timing`` and the output JSON path.
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import data  # noqa: E402
from baselines.common.ours import OurProtocolModel  # noqa: E402
from config import load_run_config  # noqa: E402
from model import GraphChangeNet  # noqa: E402

BASELINES = ("anysleep", "usleep", "cbramod", "deepsleepnet", "graphsleepnet",
             "graphsleepnet_distance", "mstgcn", "productgraphsleepnet")
RUN = REPO / "runs/exp15_final_fold0"
CONTEXT = 20
EVALUATION_BATCH = 2


class Captured(Exception):
    """Raised in place of training to hand the runner's network back.

    Parameters
    ----------
    network : nn.Module
        The network the runner built.
    """

    def __init__(self, network: nn.Module):
        super().__init__("captured")
        self.network = network


def capture(network: nn.Module, *args, **kwargs):
    """Stand-in for ``run_ours`` / ``run_graph_ours`` that stops before training.

    Parameters
    ----------
    network : nn.Module
        The network the runner built.
    *args, **kwargs
        The runner's other arguments, ignored.

    Raises
    ------
    Captured
        Always, carrying ``network``.
    """
    raise Captured(network)


def baseline_model(name: str, device: torch.device) -> tuple[nn.Module, dict]:
    """A baseline as trained under the unified configuration, with its trained weights.

    Parameters
    ----------
    name : str
        Baseline package name.
    device : torch.device
        Where to put the model.

    Returns
    -------
    model : nn.Module
        ``OurProtocolModel`` around the baseline network, in evaluation mode.
    info : dict
        How the weights loaded (missing and unexpected keys).
    """
    runner = importlib.import_module(f"baselines.{name}.runner")
    for attribute in ("run_ours", "run_graph_ours"):
        if hasattr(runner, attribute):
            setattr(runner, attribute, capture)
    settings = yaml.safe_load((REPO / "baselines/settings.yaml").read_text())
    spec = yaml.safe_load((REPO / f"baselines/{name}/ours.yaml").read_text())
    try:
        runner.run(spec, settings, REPO / "baselines/results" / name / "ours_cost", False)
    except Captured as captured:
        network = captured.network
    else:
        raise RuntimeError(f"{name}: the runner finished without building a network")
    model = OurProtocolModel(network).to(device)
    state = torch.load(REPO / "baselines/results" / name / "ours/classify_best.pt",
                       map_location=device)
    result = model.load_state_dict(state, strict=False)
    return model.eval(), {"missing_keys": len(result.missing_keys),
                          "unexpected_keys": len(result.unexpected_keys)}


def ours_model(device: torch.device) -> tuple[nn.Module, dict]:
    """GraphChangeNet (Exp 15, fold 0) with its trained weights.

    Parameters
    ----------
    device : torch.device
        Where to put the model.

    Returns
    -------
    model : nn.Module
        The trained network in evaluation mode.
    info : dict
        Empty; the weights load strictly.
    """
    cfg = load_run_config(RUN, {})
    channels, coords = data.read_montage(data.find_montage(cfg.paths.data_root))
    model = GraphChangeNet(cfg, len(channels), coords).to(device)
    model.load_state_dict(torch.load(RUN / "classify_best.pt", map_location=device))
    return model.eval(), {}


def recurrent_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Operations of the GRU and LSTM layers of one forward pass, from their sizes.

    Each step of one direction of one layer costs ``2 * gates * (input * hidden + hidden ** 2)``
    operations, with 3 gates for a GRU and 4 for an LSTM.

    Parameters
    ----------
    model : nn.Module
        The network.
    x : torch.Tensor
        Its input.

    Returns
    -------
    int
        Recurrent operations of the pass.
    """
    total = [0]

    def hook(module, inputs, outputs):
        """Add the operations of one recurrent call to the running total."""
        sequence = inputs[0]
        if module.batch_first:
            batch, steps = sequence.shape[0], sequence.shape[1]
        else:
            steps, batch = sequence.shape[0], sequence.shape[1]
        gates = 3 if isinstance(module, nn.GRU) else 4
        directions = 2 if module.bidirectional else 1
        hidden = module.hidden_size
        for layer in range(module.num_layers):
            width = module.input_size if layer == 0 else hidden * directions
            total[0] += 2 * gates * (width * hidden + hidden * hidden) * steps * batch * directions

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.GRU, nn.LSTM))]
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    return total[0]


class FFTCounter(TorchDispatchMode):
    """Counts the operations of every FFT dispatched while active.

    Attributes
    ----------
    total : float
        Accumulated operations: 5 N log2 N per complex N-point transform, half for real input.
    """

    KINDS = {"_fft_r2c": 0.5, "_fft_c2r": 0.5, "_fft_c2c": 1.0}

    def __init__(self):
        super().__init__()
        self.total = 0.0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run the operator and add its cost if it is an FFT."""
        output = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in self.KINDS:
            source = output if name == "_fft_c2r" else args[0]
            length = int(np.prod([source.shape[d] for d in args[1]]))
            transforms = source.numel() / length
            self.total += self.KINDS[name] * 5 * length * np.log2(max(length, 2)) * transforms
        return output


def fft_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Operations of the FFTs of one forward pass.

    Parameters
    ----------
    model : nn.Module
        The network.
    x : torch.Tensor
        Its input.

    Returns
    -------
    int
        FFT operations of the pass.
    """
    counter = FFTCounter()
    with torch.no_grad(), counter:
        model(x)
    return int(counter.total)


def operations(model: nn.Module, x: torch.Tensor) -> int:
    """All counted operations of one forward pass (matrix products, convolutions, recurrences, FFTs).

    Parameters
    ----------
    model : nn.Module
        The network.
    x : torch.Tensor
        Its input.

    Returns
    -------
    int
        Operations of the pass.
    """
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    return int(counter.get_total_flops()) + recurrent_flops(model, x) + fft_flops(model, x)


def peak_memory(model: nn.Module, x: torch.Tensor) -> float:
    """Peak GPU memory of one forward pass of this process, in MiB, weights included.

    Parameters
    ----------
    model : nn.Module
        The network, on the GPU.
    x : torch.Tensor
        Its input, on the GPU.

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


def clock(model: nn.Module, x: torch.Tensor, device: torch.device, repeats: int) -> dict:
    """Median, 10th and 90th percentile time of a forward pass after five warm-up passes.

    Parameters
    ----------
    model : nn.Module
        The network.
    x : torch.Tensor
        Its input.
    device : torch.device
        Device of the model.
    repeats : int
        Timed passes.

    Returns
    -------
    dict
        ``median_ms``, ``p10_ms``, ``p90_ms``.
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


def build(name: str, device: torch.device) -> tuple[nn.Module, dict]:
    """GraphChangeNet or one baseline, trained, in evaluation mode.

    Parameters
    ----------
    name : str
        ``graphchangenet`` or a baseline package name.
    device : torch.device
        Where to put the model.

    Returns
    -------
    model : nn.Module
        The network.
    info : dict
        Weight-loading notes.
    """
    return ours_model(device) if name == "graphchangenet" else baseline_model(name, device)


def static(out: Path) -> None:
    """Parameters, operations and peak memory of every model on the same input.

    Parameters
    ----------
    out : Path
        Output JSON path; existing entries of other models are kept.
    """
    device = torch.device("cuda")
    result = json.loads(out.read_text()) if out.exists() else {}
    samples = 30000
    for name in ("graphchangenet",) + BASELINES:
        try:
            model, info = build(name, device)
            one = torch.randn(1, CONTEXT, 83, samples, device=device)
            batch = torch.randn(EVALUATION_BATCH, CONTEXT, 83, samples, device=device)
            run_operations = operations(model, one)
            result[name] = {**info,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "gflops_per_scored_epoch_as_evaluated": run_operations / 1e9,
                            "gflops_per_epoch_amortised": run_operations / CONTEXT / 1e9,
                            "peak_mib_one_run": peak_memory(model, one),
                            "peak_mib_evaluation_batch": peak_memory(model, batch)}
            del model, one, batch
        except Exception as error:
            result[name] = {"error": f"{type(error).__name__}: {error}"}
        torch.cuda.empty_cache()
        print(name, json.dumps(result[name]), flush=True)
        out.write_text(json.dumps(result, indent=1))


def timing(out: Path) -> None:
    """Latency of one 20-epoch run and evaluation-batch throughput of every model, GPU and CPU.

    Parameters
    ----------
    out : Path
        Output JSON path; existing entries of other models are kept.
    """
    result = json.loads(out.read_text()) if out.exists() else {}
    for name in ("graphchangenet",) + BASELINES:
        entry = {}
        for device_name, repeats in (("cuda", 20), ("cpu", 3)):
            device = torch.device(device_name)
            try:
                model, _ = build(name, device)
                one = torch.randn(1, CONTEXT, 83, 30000, device=device)
                batch = torch.randn(EVALUATION_BATCH, CONTEXT, 83, 30000, device=device)
                single = clock(model, one, device, repeats)
                batched = clock(model, batch, device, repeats)
                entry[device_name] = {"run_ms": single, "evaluation_batch_ms": batched,
                                      "scored_epochs_per_second":
                                          EVALUATION_BATCH / batched["median_ms"] * 1000}
                del model, one, batch
            except Exception as error:
                entry[device_name] = {"error": f"{type(error).__name__}: {error}"}
            if device_name == "cuda":
                torch.cuda.empty_cache()
        result[name] = entry
        print(name, json.dumps(entry), flush=True)
        out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"static": static, "timing": timing}[sys.argv[1]](Path(sys.argv[2]))
