"""AnySleep networks built from the unmodified upstream code.

The upstream repository is imported from ``upstream/examples/anphy`` through
``sys.path``; nothing in ``upstream/`` is changed. :func:`released_network` is
the released model exactly as ``predict_anphy.py`` builds it.
:class:`AnySleepBackbone` keeps the upstream ``forward`` and module names but
takes the kernel size and the pooling width as arguments, which upstream
hard-codes to 9 and to 128 Hz times 30 s. :class:`AnySleepRuns` presents it
with our ``[B, S, C, T]`` runs of epochs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / "upstream"
EXAMPLES = UPSTREAM / "examples" / "anphy"


def upstream_module(name: str) -> ModuleType:
    """Import a module from ``upstream/examples/anphy`` without changing it.

    ``predict_anphy.py`` imports ``anysleep_no_hydra`` by bare name, so the
    folder goes on ``sys.path`` rather than being loaded file by file.

    Parameters
    ----------
    name : str
        Module name, ``anysleep_no_hydra`` or ``predict_anphy``.

    Returns
    -------
    ModuleType
        The imported module.
    """
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return __import__(name)


def released_network(checkpoint: Path, settings: dict) -> nn.Module:
    """The released AnySleep model with one of its checkpoints loaded.

    Parameters
    ----------
    checkpoint : Path
        ``anysleep-run<k>.pth``.
    settings : dict
        ``depth``, ``complexity``, ``scale``, ``hidden_size`` and
        ``sleep_stage_frequency``, passed to the upstream constructor.

    Returns
    -------
    nn.Module
        Upstream ``AnySleep``, mapping ``(batch, time, channels)`` at 128 Hz
        to logits ``(batch, epochs, 5)``.
    """
    upstream = upstream_module("anysleep_no_hydra")
    return upstream.AnySleep(path=str(checkpoint), depth=settings["depth"],
                             complexity=settings["complexity"], scale=settings["scale"],
                             hidden_size=settings["hidden_size"],
                             sleep_stage_frequency=settings["sleep_stage_frequency"])


def upstream_initialisation(module: nn.Module) -> None:
    """The weight initialisation of upstream's training code, applied to one module.

    A copy of ``_weights_init`` in ``upstream/base/model/base_model.py``
    (lines 16-53). That file imports Hydra, which is not installed here. The
    standalone model never calls it, but the released models were trained
    with it. Linear and convolution weights get Kaiming normal (fan out,
    ReLU), linear biases zero, batch norms weight 1 and bias 0.

    Parameters
    ----------
    module : nn.Module
        A submodule visited by ``nn.Module.apply``.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def backbone_class() -> type:
    """Build :class:`AnySleepBackbone` on top of the upstream ``AnySleep`` class.

    The class is created at call time so that importing this module does not
    import upstream code.

    Returns
    -------
    type
        A subclass of upstream ``AnySleep`` whose constructor takes the kernel
        size and the pooling width.
    """
    upstream = upstream_module("anysleep_no_hydra")

    class AnySleepBackbone(upstream.AnySleep):
        """Upstream AnySleep with a configurable kernel size and pooling width.

        Its modules are the upstream blocks (``USleepEncoderBlock``,
        ``USleepDecoderBlock``, ``SkipConnectionBlock``, ``SegmentClassifier``)
        under the upstream attribute names, and ``forward`` is inherited
        unchanged. With depth 12, kernel 9 and pooling width 3840, the released
        checkpoints load into it.

        Parameters
        ----------
        depth : int
            Encoder and decoder blocks.
        complexity, scale : float
            Filter multiplier and growth per block (upstream ``calc_filters``).
        hidden_size : int
            Hidden units of each channel-attention MLP.
        kernel_size : int
            Convolution kernel of every encoder block, decoder block and the
            connector.
        pool_samples : int
            Average-pooling width of the segment classifier, in samples: one
            prediction per ``pool_samples`` input samples.
        """

        def __init__(self, depth: int, complexity: float, scale: float, hidden_size: int,
                     kernel_size: int, pool_samples: int):
            nn.Module.__init__(self)
            encoder_filters, decoder_filters = upstream.calc_filters(1, depth, complexity, scale)
            self.encoders = nn.ModuleList([
                upstream.USleepEncoderBlock(encoder_filters[i - 1], encoder_filters[i],
                                            kernel_size)
                for i in range(1, len(encoder_filters))])
            self.connector = nn.Sequential(
                nn.Conv1d(encoder_filters[-1], decoder_filters[0], kernel_size, padding="same"),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(decoder_filters[0]))
            self.decoders = nn.ModuleList([
                upstream.USleepDecoderBlock(decoder_filters[i - 1], decoder_filters[i],
                                            kernel_size)
                for i in range(1, len(decoder_filters))])
            self.segment_classifier = upstream.SegmentClassifier(pool_samples,
                                                                 decoder_filters[-1])
            self.skip_connections = nn.ModuleList(
                [upstream.SkipConnectionBlock(encoder_filters[i], hidden_size)
                 for i in range(1, len(encoder_filters))]
                + [upstream.SkipConnectionBlock(decoder_filters[0], hidden_size)])
            self.apply(upstream_initialisation)

    return AnySleepBackbone


class AnySleepRuns(nn.Module):
    """AnySleep on our runs of epochs, one prediction per epoch.

    Our loaders give ``[B, S, C, T]``: ``S`` consecutive epochs of ``T``
    samples on ``C`` electrodes. Upstream AnySleep reads a continuous
    recording ``(B, time, channels)``, so the ``S`` epochs are laid end to end
    (``time = S * T``) and the segment classifier pools every
    ``pool_samples`` samples into one prediction. With ``pool_samples = T``
    that is one prediction per epoch.

    Parameters
    ----------
    depth : int
        Encoder and decoder blocks.
    complexity, scale : float
        Filter multiplier and growth per block.
    hidden_size : int
        Hidden units of each channel-attention MLP.
    kernel_size : int
        Convolution kernel size.
    pool_samples : int
        Samples per prediction; must equal the samples per epoch ``T``.
    """

    def __init__(self, depth: int, complexity: float, scale: float, hidden_size: int,
                 kernel_size: int, pool_samples: int):
        super().__init__()
        self.pool_samples = pool_samples
        self.backbone = backbone_class()(depth, complexity, scale, hidden_size, kernel_size,
                                         pool_samples)
        self.classifier = self.backbone.segment_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of every epoch of every run.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, S, C, T]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, S, 5]``.

        Raises
        ------
        ValueError
            If an epoch is not ``pool_samples`` samples long.
        """
        batch, n_epochs, n_channels, samples = x.shape
        if samples != self.pool_samples:
            raise ValueError(f"epochs of {samples} samples, but the classifier pools "
                             f"{self.pool_samples}")
        recording = x.permute(0, 1, 3, 2).reshape(batch, n_epochs * samples, n_channels)
        return self.backbone(recording)
