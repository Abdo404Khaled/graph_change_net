from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

UPSTREAM_MODULE = Path(__file__).resolve().parents[1] / "anysleep" / "upstream" / "examples" / \
    "usleep_no_hydra.py"


@cache
def anysleep_usleep() -> ModuleType:
    """The AnySleep authors' stand-alone PyTorch U-Sleep, loaded read-only from its file.

    ``examples/usleep_no_hydra.py`` of the AnySleep repository holds the
    official U-Sleep architecture (filter widths, kernels, padding side,
    centre crop, concatenation order and segment classifier of
    ``perslev/U-Time``) with no dependency but torch and numpy. It is loaded
    from its path so that nothing is added to ``sys.path`` and the file is
    never copied or edited.

    Returns
    -------
    ModuleType
        The module, with ``USleep``, ``USleepEncoderBlock``,
        ``USleepDecoderBlock``, ``SegmentClassifier`` and ``calc_filters``.

    Raises
    ------
    FileNotFoundError
        If the AnySleep clone is missing.
    """
    if not UPSTREAM_MODULE.exists():
        raise FileNotFoundError(f"{UPSTREAM_MODULE} is missing; clone dslaborg/AnySleep there")
    spec = importlib.util.spec_from_file_location("anysleep_usleep_no_hydra", UPSTREAM_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pretrained_usleep(checkpoint: Path, depth: int, complexity: float, scale: float,
                      sleep_stage_frequency: int) -> nn.Module:
    """The AnySleep authors' U-Sleep with one of their released checkpoints.

    Parameters
    ----------
    checkpoint : Path
        ``usleep-run{1,2,3}.pth``, read with ``weights_only=True``.
    depth : int
        Encoder and decoder blocks (12 in the release).
    complexity : float
        Filter multiplier, sqrt(1.67) in the release.
    scale : float
        Filter growth per level, sqrt(2) in the release.
    sleep_stage_frequency : int
        Predictions per 30 s epoch (1 for standard staging).

    Returns
    -------
    nn.Module
        Their ``USleep`` in evaluation mode: input ``[B, time, 2]`` at 128 Hz
        (EEG, EOG), output logits ``[B, epochs, 5]``.
    """
    module = anysleep_usleep()
    network = module.USleep(path=str(checkpoint), depth=depth, complexity=complexity,
                            scale=scale, sleep_stage_frequency=sleep_stage_frequency)
    return network.eval()


def anysleep_initialisation(module: nn.Module) -> None:
    """The weight initialisation the AnySleep authors trained their U-Sleep with.

    It reproduces ``_weights_init`` of ``base/model/base_model.py`` in their
    repository for the layer types U-Sleep has: Kaiming-normal convolution
    weights (fan-out, ReLU gain) with PyTorch's default biases, and
    BatchNorm weights 1 and biases 0. That file imports Hydra, so it is
    reproduced here rather than imported.

    Parameters
    ----------
    module : nn.Module
        Layer to initialise, as passed by ``nn.Module.apply``.
    """
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class USleepNetwork(nn.Module):
    """U-Sleep built from the AnySleep blocks, with the input size and epoch length as settings.

    The encoder blocks, decoder blocks, filter widths and segment classifier
    are the AnySleep classes, so the computation is the official one. Only
    the values the port hard-codes are exposed: the number of input channels
    (2 there), the convolution kernel of the encoder, connector and decoder
    (9 samples there) and the width of the segment classifier's average
    pooling (3840 = 30 s at 128 Hz there). With 2 channels, kernel 9 and
    pooling 3840 the parameter names and shapes equal the released
    checkpoints, which load into this class unchanged.

    Parameters
    ----------
    n_channels : int
        Input channels.
    depth : int
        Encoder and decoder blocks.
    kernel_size : int
        Kernel of every encoder, connector and decoder convolution, in samples.
    complexity : float
        Filter multiplier.
    scale : float
        Filter growth per level.
    pool_samples : int
        Samples averaged per output, the epoch length in samples.
    """

    def __init__(self, n_channels: int, depth: int, kernel_size: int, complexity: float,
                 scale: float, pool_samples: int):
        super().__init__()
        blocks = anysleep_usleep()
        encoder_widths, decoder_widths = blocks.calc_filters(n_channels, depth, complexity, scale)
        encoder_widths = [int(width) for width in encoder_widths]
        decoder_widths = [int(width) for width in decoder_widths]
        self.encoders = nn.ModuleList(
            blocks.USleepEncoderBlock(encoder_widths[i - 1], encoder_widths[i], kernel_size)
            for i in range(1, len(encoder_widths)))
        self.connector = nn.Sequential(
            nn.Conv1d(encoder_widths[-1], decoder_widths[0], kernel_size, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(decoder_widths[0]))
        self.decoders = nn.ModuleList(
            blocks.USleepDecoderBlock(decoder_widths[i - 1], decoder_widths[i], kernel_size)
            for i in range(1, len(decoder_widths)))
        self.segment_classifier = blocks.SegmentClassifier(1, decoder_widths[-1])
        self.segment_classifier.segment_classifier[2] = nn.AvgPool1d(pool_samples,
                                                                     stride=pool_samples)
        self.pool_samples = pool_samples
        self.apply(anysleep_initialisation)

    @property
    def classifier(self) -> nn.Module:
        """U-Sleep's segment classifier, the head for our optimiser's parameter groups.

        Returns
        -------
        nn.Module
            The segment classifier (1x1 conv, tanh, average pooling, 1x1 conv,
            ELU, 1x1 conv).
        """
        return self.segment_classifier

    def forward_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of a continuous signal.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, C, samples]``, samples a multiple of ``pool_samples``.

        Returns
        -------
        torch.Tensor
            ``[B, n_stages, samples // pool_samples]``.
        """
        residuals = []
        for encoder in self.encoders:
            residual, x = encoder(x)
            residuals.append(residual)
        x = self.connector(x)
        for decoder in self.decoders:
            x = decoder(x, residuals.pop())
        return self.segment_classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of every epoch of every run.

        The epochs of a run are concatenated in time, so the network sees the
        run as one continuous signal, as U-Sleep sees a night.

        Parameters
        ----------
        x : torch.Tensor
            Runs of epochs ``[B, S, C, T]`` with ``T == pool_samples``.

        Returns
        -------
        torch.Tensor
            ``[B, S, n_stages]``.

        Raises
        ------
        ValueError
            If the epoch length differs from the pooling width.
        """
        batch, n_epochs, n_channels, n_samples = x.shape
        if n_samples != self.pool_samples:
            raise ValueError(f"epochs of {n_samples} samples, pooling over {self.pool_samples}")
        signal = x.permute(0, 2, 1, 3).reshape(batch, n_channels, n_epochs * n_samples)
        return self.forward_signal(signal).transpose(1, 2)
