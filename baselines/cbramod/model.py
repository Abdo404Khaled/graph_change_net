from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Port of ``upstream/models/cbramod.py:PatchEmbedding`` (time-frequency patch encoder + ACPE).

    The time branch is three convolution blocks over each patch (25 filters,
    kernels 49/3/3, strides 25/1/1, GroupNorm, GELU), whose ``25 x 8``
    outputs form the ``d_model = 200`` features; the frequency branch maps
    the magnitudes of the patch's ``norm='forward'`` FFT through
    ``Linear(101, d_model)``; the asymmetric conditional positional encoding
    is a depthwise ``(19, 7)`` convolution over the (channel, patch) grid.
    Module and parameter names are upstream's, so the checkpoint loads
    strictly.

    Parameters
    ----------
    in_dim : int
        Points per patch (size of the fixed all-zero mask token).
    d_model : int
        Patch embedding width.
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.spectral_bins = in_dim // 2 + 1
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7),
                      stride=(1, 1), padding=(9, 3), groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25),
                      padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Embed every patch and add the positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Patches ``[B, C, n, in_dim]``.
        mask : torch.Tensor, optional
            ``[B, C, n]`` with 1 where a patch is replaced by the mask token.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, d_model]``.
        """
        batch, n_channels, n_patches, patch_samples = x.shape
        if mask is None:
            masked = x
        else:
            masked = x.clone()
            masked[mask == 1] = self.mask_encoding
        masked = masked.contiguous().view(batch, 1, n_channels * n_patches, patch_samples)
        embedding = self.proj_in(masked)
        embedding = embedding.permute(0, 2, 1, 3).contiguous().view(
            batch, n_channels, n_patches, self.d_model)
        flat = masked.contiguous().view(batch * n_channels * n_patches, patch_samples)
        spectrum = torch.fft.rfft(flat, dim=-1, norm="forward")[:, :self.spectral_bins]
        spectrum = torch.abs(spectrum).contiguous().view(batch, n_channels, n_patches,
                                                         self.spectral_bins)
        embedding = embedding + self.spectral_proj(spectrum)
        position = self.positional_encoding(embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return embedding + position


class CrissCrossLayer(nn.Module):
    """Port of ``upstream/models/criss_cross_transformer.py:TransformerEncoderLayer``.

    Pre-norm block. The first half of the features goes through S-attention
    (over the channels at each patch position), the second half through
    T-attention (over the patches of each channel), each with
    ``nhead // 2`` heads; the halves are concatenated, then a GELU
    feed-forward block follows. Dropout sits where upstream puts it.

    Parameters
    ----------
    d_model : int
        Feature width.
    nhead : int
        Heads in total (half spatial, half temporal).
    dim_feedforward : int
        Feed-forward width.
    dropout : float
        Dropout in attention and feed-forward.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout,
                                                 bias=True, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout,
                                                 bias=True, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Criss-cross attention block, then feed-forward block, both residual.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, C, n, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, d_model]``.
        """
        x = x + self.attention_block(self.norm1(x))
        return x + self.feedforward_block(self.norm2(x))

    def attention_block(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial attention on one half of the features, temporal on the other.

        Parameters
        ----------
        x : torch.Tensor
            Normalised ``[B, C, n, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, d_model]``.
        """
        batch, n_channels, n_patches, width = x.shape
        half = width // 2
        spatial = x[:, :, :, :half].transpose(1, 2).contiguous().view(
            batch * n_patches, n_channels, half)
        temporal = x[:, :, :, half:].contiguous().view(batch * n_channels, n_patches, half)
        spatial = self.self_attn_s(spatial, spatial, spatial, need_weights=False)[0]
        spatial = spatial.contiguous().view(batch, n_patches, n_channels, half).transpose(1, 2)
        temporal = self.self_attn_t(temporal, temporal, temporal, need_weights=False)[0]
        temporal = temporal.contiguous().view(batch, n_channels, n_patches, half)
        return self.dropout1(torch.concat((spatial, temporal), dim=3))

    def feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        """GELU feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Normalised ``[B, C, n, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, d_model]``.
        """
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class CrissCrossEncoder(nn.Module):
    """Port of ``upstream/models/criss_cross_transformer.py:TransformerEncoder``.

    ``n_layer`` deep copies of one layer, as upstream's ``_get_clones``
    makes them (so every layer starts from the same initial weights before
    the checkpoint is loaded), with no final norm.

    Parameters
    ----------
    layer : CrissCrossLayer
        Layer to copy.
    n_layer : int
        Number of layers.
    """

    def __init__(self, layer: CrissCrossLayer, n_layer: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layers in turn.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, C, n, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, d_model]``.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class CBraMod(nn.Module):
    """Port of ``upstream/models/cbramod.py:CBraMod`` (backbone and reconstruction head).

    Built in upstream's order (patch embedding, one encoder layer cloned
    ``n_layer`` times, ``proj_out``), then every ``nn.Linear`` is
    re-initialised with Kaiming-normal as upstream's ``_weights_init`` does,
    so a seeded run draws the same random numbers before the head is built.

    Parameters
    ----------
    in_dim : int
        Points per patch.
    out_dim : int
        Points per reconstructed patch.
    d_model : int
        Feature width.
    dim_feedforward : int
        Feed-forward width.
    seq_len : int
        Patches per sample (unused by the computation, kept for the signature).
    n_layer : int
        Encoder layers.
    nhead : int
        Heads per layer.
    dropout : float
        Dropout of the encoder layers.
    """

    def __init__(self, in_dim: int, out_dim: int, d_model: int, dim_feedforward: int,
                 seq_len: int, n_layer: int, nhead: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.patch_embedding = PatchEmbedding(in_dim, d_model)
        self.encoder = CrissCrossEncoder(CrissCrossLayer(d_model, nhead, dim_feedforward,
                                                         dropout), n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(kaiming_linear)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Patch features (or reconstructed patches while ``proj_out`` is in place).

        Parameters
        ----------
        x : torch.Tensor
            ``[B, C, n, in_dim]``.
        mask : torch.Tensor, optional
            ``[B, C, n]`` mask of the pretraining objective.

        Returns
        -------
        torch.Tensor
            ``[B, C, n, out_dim]``.
        """
        return self.proj_out(self.encoder(self.patch_embedding(x, mask)))


def kaiming_linear(module: nn.Module) -> None:
    """Upstream's ``_weights_init``: Kaiming-normal (fan-out, ReLU gain) on linear weights.

    Parameters
    ----------
    module : nn.Module
        Module visited by ``nn.Module.apply``.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


def rescaled_kernel(weight: torch.Tensor, factor: int) -> torch.Tensor:
    """Resample a pretrained temporal convolution kernel to a sampling rate ``factor`` times higher.

    The pretrained kernel has ``K`` taps; the new one has ``factor * K``
    taps over the same duration and with the same centre. Tap ``m`` sits at
    pretrained position ``(m - centre_new) / factor + centre_old`` and takes
    the linear interpolation of the pretrained taps there (zero outside).
    Dividing by ``factor`` keeps the response to a slow signal unchanged,
    since the new kernel sums ``factor`` times as many samples.

    Parameters
    ----------
    weight : torch.Tensor
        Pretrained kernel ``[out, in, 1, K]``.
    factor : int
        Ratio of the new to the pretrained sampling rate.

    Returns
    -------
    torch.Tensor
        Kernel ``[out, in, 1, factor * K]``.
    """
    taps = weight.shape[-1]
    new_taps = factor * taps
    centre, new_centre = (taps - 1) / 2, (new_taps - 1) / 2
    position = (np.arange(new_taps) - new_centre) / factor + centre
    flat = weight.detach().reshape(-1, taps).cpu().numpy()
    resampled = np.stack([np.interp(position, np.arange(taps), row, left=0.0, right=0.0)
                          for row in flat]) / factor
    return torch.from_numpy(resampled.astype(np.float32)).reshape(*weight.shape[:-1], new_taps)


def stretch_patch_embedding(pretrained: PatchEmbedding, patch_samples: int) -> PatchEmbedding:
    """CBraMod's patch encoder for 1 s patches recorded at a multiple of 200 Hz.

    A patch stays 1 s, as in pretraining, but holds ``patch_samples`` points.
    The first convolution is stretched by ``factor = patch_samples /
    in_dim`` (kernel, stride and padding times ``factor``, centred, weights
    from :func:`rescaled_kernel`), so the time branch still yields 8
    positions of 25 filters, i.e. ``d_model`` features. The frequency branch
    keeps the pretrained ``Linear(101, d_model)`` and reads the first 101
    bins of the ``patch_samples``-point FFT: with ``norm='forward'`` these
    are 0-100 Hz at 1 Hz, the grid and amplitude scale the pretrained branch
    saw at 200 Hz. The other blocks and ACPE are the pretrained ones.

    Parameters
    ----------
    pretrained : PatchEmbedding
        Patch encoder holding the pretrained weights.
    patch_samples : int
        Points per 1 s patch at the recording rate.

    Returns
    -------
    PatchEmbedding
        A new encoder; the pretrained one is left as it is.

    Raises
    ------
    ValueError
        If ``patch_samples`` is not a multiple of the pretrained patch length.
    """
    in_dim = pretrained.mask_encoding.numel()
    if patch_samples % in_dim:
        raise ValueError(f"patch_samples {patch_samples} is not a multiple of {in_dim}")
    factor = patch_samples // in_dim
    stretched = copy.deepcopy(pretrained)
    stretched.mask_encoding = nn.Parameter(torch.zeros(patch_samples), requires_grad=False)
    stretched.spectral_bins = pretrained.spectral_bins
    first = pretrained.proj_in[0]
    kernel = first.kernel_size[1] * factor
    conv = nn.Conv2d(first.in_channels, first.out_channels, kernel_size=(1, kernel),
                     stride=(1, first.stride[1] * factor), padding=(0, (kernel - 1) // 2))
    with torch.no_grad():
        conv.weight.copy_(rescaled_kernel(first.weight, factor))
        conv.bias.copy_(first.bias)
    stretched.proj_in[0] = conv
    return stretched


class SequenceModel(nn.Module):
    """Port of ``upstream/models/model_for_isruc.py:Model``, generalised in size.

    The pretrained backbone with an identity in place of its reconstruction
    head; the flatten ``Linear(C * n * d_model, head_hidden)`` + GELU per
    epoch; a bidirectional ``nn.TransformerEncoder`` over the epochs of a
    run (no mask, no positional encoding); ``Linear(head_hidden,
    n_classes)``. Modules are built in upstream's order. With
    ``patch_samples`` above the pretrained patch length the patch encoder is
    replaced by :func:`stretch_patch_embedding`; ``channel_order`` permutes
    the electrodes before the backbone (identity for the native setting).

    Parameters
    ----------
    weights : Path
        Pretrained checkpoint.
    backbone : dict
        :class:`CBraMod` arguments.
    channel_order : np.ndarray
        Input electrode index at each backbone position ``[C]``.
    n_patches : int
        Patches per epoch.
    patch_samples : int
        Points per patch.
    head_hidden : int
        Width of the epoch token and of the sequence transformer.
    sequence_heads : int
        Heads of the sequence transformer.
    sequence_feedforward : int
        Feed-forward width of the sequence transformer.
    sequence_layers : int
        Layers of the sequence transformer.
    n_classes : int
        Number of stages.
    """

    def __init__(self, weights: Path, backbone: dict, channel_order: np.ndarray, n_patches: int,
                 patch_samples: int, head_hidden: int, sequence_heads: int,
                 sequence_feedforward: int, sequence_layers: int, n_classes: int):
        super().__init__()
        self.n_patches, self.patch_samples = n_patches, patch_samples
        self.d_model = backbone["d_model"]
        self.register_buffer("channel_order", torch.as_tensor(channel_order, dtype=torch.long))
        self.backbone = CBraMod(**backbone)
        self.backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        if patch_samples != backbone["in_dim"]:
            self.backbone.patch_embedding = stretch_patch_embedding(
                self.backbone.patch_embedding, patch_samples)
        self.backbone.proj_out = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(len(channel_order) * n_patches * self.d_model, head_hidden),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=head_hidden, nhead=sequence_heads, dim_feedforward=sequence_feedforward,
            batch_first=True, activation=F.gelu, norm_first=True)
        self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=sequence_layers,
                                                      enable_nested_tensor=False)
        self.classifier = nn.Linear(head_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of every epoch of every run.

        Parameters
        ----------
        x : torch.Tensor
            Runs of epochs ``[B, S, C, n_patches * patch_samples]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, S, n_classes]``.
        """
        batch, n_epochs, n_channels, _ = x.shape
        x = x.index_select(2, self.channel_order)
        x = x.contiguous().view(batch * n_epochs, n_channels, self.n_patches, self.patch_samples)
        features = self.backbone(x).contiguous().view(
            batch, n_epochs, n_channels * self.n_patches * self.d_model)
        return self.classifier(self.sequence_encoder(self.head(features)))


def nearest_neighbour_chain(coords: np.ndarray) -> np.ndarray:
    """Electrode order that walks the scalp, each step to the nearest unvisited electrode.

    ACPE mixes the electrodes within 9 indices of each other, so neighbouring
    indices should be neighbouring electrodes. The montage file's order is
    not: it lists the 10-20 electrodes left and right in turn and then the
    extended ones, so consecutive indices are 8.5 cm apart on average on
    ANPHY's co-registered positions. The greedy chain starts at the largest
    anterior coordinate and always steps to the nearest electrode not yet
    visited (ties to the lower index), which brings consecutive indices to
    3.4 cm and a 19-index window from 10.2 to 8.4 cm of mean pairwise
    distance. The chain is fixed by the coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Electrode coordinates ``[C, 3]`` (x anterior, y left, z up).

    Returns
    -------
    np.ndarray
        Input electrode index at each backbone position ``[C]``.
    """
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    order = [int(np.argmax(coords[:, 0]))]
    unvisited = np.ones(len(coords), dtype=bool)
    unvisited[order[0]] = False
    while unvisited.any():
        candidates = np.flatnonzero(unvisited)
        step = int(candidates[np.argmin(distance[order[-1], candidates])])
        order.append(step)
        unvisited[step] = False
    return np.array(order)
