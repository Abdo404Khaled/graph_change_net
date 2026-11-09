"""PyTorch port of DeepSleepNet (Supratak et al., 2017), following ``upstream/deepsleep/model.py``.

The port reproduces the TensorFlow graph layer by layer: "SAME" padding on
every convolution and pooling, convolutions without bias followed by batch
normalisation (moving-average decay 0.999, so momentum 0.001) and ReLU, He
truncated-normal weights, two independent two-layer LSTM stacks with
peephole connections for the forward and backward directions (TensorFlow's
``static_bidirectional_rnn`` over ``MultiRNNCell``), dropout on every LSTM
layer's output, and a fully connected shortcut added to the LSTM output.
First-layer kernel sizes and strides are given in seconds and scaled by the
sampling rate, as the paper prescribes (Fs/2 and Fs/16, 4 Fs and Fs/2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

State = tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]


def same_padding(length: int, kernel: int, stride: int) -> tuple[int, int]:
    """Left and right padding of TensorFlow's ``padding="SAME"``.

    Parameters
    ----------
    length : int
        Input length.
    kernel : int
        Kernel or pooling size.
    stride : int
        Stride.

    Returns
    -------
    tuple of int
        Padding before and after, the extra sample going after.
    """
    output = math.ceil(length / stride)
    total = max((output - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


def same_length(length: int, stride: int) -> int:
    """Output length of a "SAME" convolution or pooling.

    Parameters
    ----------
    length : int
        Input length.
    stride : int
        Stride.

    Returns
    -------
    int
        ``ceil(length / stride)``.
    """
    return math.ceil(length / stride)


def he_truncated_normal_(weight: torch.Tensor, fan_in: int) -> None:
    """Upstream's weight initialisation, in place.

    ``deepsleep/nn.py`` draws every convolution and fully connected weight
    from a normal distribution truncated at two standard deviations, with
    standard deviation ``sqrt(2 / fan_in)`` (He et al., 2015).

    Parameters
    ----------
    weight : torch.Tensor
        Weight to fill.
    fan_in : int
        Inputs per output unit (input channels times kernel size).
    """
    deviation = math.sqrt(2.0 / fan_in)
    nn.init.trunc_normal_(weight, 0.0, deviation, -2.0 * deviation, 2.0 * deviation)


def batch_norm(n_features: int, settings: dict) -> nn.BatchNorm1d:
    """Batch normalisation initialised as upstream's ``batch_norm_new``.

    Parameters
    ----------
    n_features : int
        Channels or features.
    settings : dict
        ``epsilon``, ``momentum`` (one minus the TensorFlow decay) and
        ``gamma_std`` (gamma is drawn from a normal around 1).

    Returns
    -------
    nn.BatchNorm1d
        The layer.
    """
    layer = nn.BatchNorm1d(n_features, eps=settings["epsilon"], momentum=settings["momentum"])
    nn.init.normal_(layer.weight, 1.0, settings["gamma_std"])
    nn.init.zeros_(layer.bias)
    return layer


class ConvBlock(nn.Module):
    """One-dimensional convolution without bias, batch normalisation and ReLU, "SAME" padded.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Filters.
    kernel : int
        Kernel size in samples.
    stride : int
        Stride in samples.
    norm : dict
        Batch-normalisation settings (:func:`batch_norm`).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int,
                 norm: dict):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel, stride, bias=False)
        self.normalisation = batch_norm(out_channels, norm)
        he_truncated_normal_(self.convolution.weight, in_channels * kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, in_channels, L]``.

        Returns
        -------
        torch.Tensor
            ``[N, out_channels, ceil(L / stride)]``.
        """
        x = F.pad(x, same_padding(x.shape[-1], self.kernel, self.stride))
        return F.relu(self.normalisation(self.convolution(x)))


class SamePool(nn.Module):
    """Max pooling with size equal to stride and "SAME" padding (padded values never win).

    Parameters
    ----------
    size : int
        Pooling size and stride.
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, C, L]``.

        Returns
        -------
        torch.Tensor
            ``[N, C, ceil(L / size)]``.
        """
        x = F.pad(x, same_padding(x.shape[-1], self.size, self.size), value=float("-inf"))
        return F.max_pool1d(x, self.size, self.size)


class FeatureBranch(nn.Module):
    """One CNN of the representation-learning part (``model.py:73-110`` or ``114-151``).

    conv1 (large stride), max-pool, dropout, three convolutions of stride 1,
    max-pool, flatten.

    Parameters
    ----------
    in_channels : int
        Input channels.
    first_filters : int
        Filters of the first convolution.
    deep_filters : int
        Filters of the three later convolutions.
    first_kernel : int
        First kernel size in samples.
    first_stride : int
        First stride in samples.
    first_pool : int
        First pooling size and stride.
    deep_kernel : int
        Kernel size of the three later convolutions.
    last_pool : int
        Last pooling size and stride.
    dropout : float
        Dropout probability after the first pooling.
    norm : dict
        Batch-normalisation settings.
    """

    def __init__(self, in_channels: int, first_filters: int, deep_filters: int, first_kernel: int,
                 first_stride: int, first_pool: int, deep_kernel: int, last_pool: int,
                 dropout: float, norm: dict):
        super().__init__()
        self.first_stride, self.first_pool_size, self.last_pool_size = (first_stride, first_pool,
                                                                         last_pool)
        self.deep_filters = deep_filters
        self.first = ConvBlock(in_channels, first_filters, first_kernel, first_stride, norm)
        self.first_pool = SamePool(first_pool)
        self.dropout = nn.Dropout(dropout)
        self.deep = nn.Sequential(ConvBlock(first_filters, deep_filters, deep_kernel, 1, norm),
                                  ConvBlock(deep_filters, deep_filters, deep_kernel, 1, norm),
                                  ConvBlock(deep_filters, deep_filters, deep_kernel, 1, norm))
        self.last_pool = SamePool(last_pool)

    def output_size(self, n_samples: int) -> int:
        """Length of the flattened output for an epoch of ``n_samples``.

        Parameters
        ----------
        n_samples : int
            Samples per epoch.

        Returns
        -------
        int
            ``deep_filters`` times the final length.
        """
        length = same_length(n_samples, self.first_stride)
        length = same_length(same_length(length, self.first_pool_size), self.last_pool_size)
        return self.deep_filters * length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features of a batch of epochs.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, C, L]``.

        Returns
        -------
        torch.Tensor
            ``[N, output_size(L)]``.
        """
        x = self.dropout(self.first_pool(self.first(x)))
        return self.last_pool(self.deep(x)).flatten(1)


class DeepFeatureNet(nn.Module):
    """Representation learning: the small- and large-filter CNNs, concatenated, then dropout.

    Parameters
    ----------
    in_channels : int
        Input channels (1 natively).
    n_samples : int
        Samples per epoch.
    sampling_rate : float
        Hz; first kernels and strides are ``int(sampling_rate * seconds)``.
    first_filters, deep_filters : int
        Filters of the first and of the later convolutions.
    small, large : dict
        Per branch: ``first_kernel_s``, ``first_stride_s`` (seconds),
        ``first_pool``, ``deep_kernel`` and ``last_pool`` (samples).
    dropout : float
        Dropout probability.
    norm : dict
        Batch-normalisation settings.
    """

    def __init__(self, in_channels: int, n_samples: int, sampling_rate: float,
                 first_filters: int, deep_filters: int, small: dict, large: dict,
                 dropout: float, norm: dict):
        super().__init__()

        def branch(spec: dict) -> FeatureBranch:
            """Build one branch from its settings.

            Parameters
            ----------
            spec : dict
                Branch settings.

            Returns
            -------
            FeatureBranch
                The branch.
            """
            return FeatureBranch(in_channels, first_filters, deep_filters,
                                 int(sampling_rate * spec["first_kernel_s"]),
                                 int(sampling_rate * spec["first_stride_s"]), spec["first_pool"],
                                 spec["deep_kernel"], spec["last_pool"], dropout, norm)

        self.small = branch(small)
        self.large = branch(large)
        self.dropout = nn.Dropout(dropout)
        self.n_samples = n_samples
        self.feature_size = self.small.output_size(n_samples) + self.large.output_size(n_samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenated features of a batch of epochs.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, C, L]``.

        Returns
        -------
        torch.Tensor
            ``[N, feature_size]``.
        """
        return self.dropout(torch.cat([self.small(x), self.large(x)], dim=1))


class PretrainNet(nn.Module):
    """Pretraining model: DeepFeatureNet stacked with its own softmax layer (paper Alg. 1, line 2).

    Parameters
    ----------
    features : DeepFeatureNet
        The CNNs that fine-tuning later takes over.
    n_classes : int
        Stages.
    """

    def __init__(self, features: DeepFeatureNet, n_classes: int):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(features.feature_size, n_classes)
        he_truncated_normal_(self.classifier.weight, features.feature_size)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage logits of single epochs.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, C, L]``.

        Returns
        -------
        torch.Tensor
            ``[N, n_classes]``.
        """
        return self.classifier(self.features(x))


class PeepholeLSTM(nn.Module):
    """One layer of TensorFlow's ``LSTMCell(use_peepholes=True)`` unrolled over time.

    Gates follow TensorFlow's order (input, candidate, forget, output) and
    equations: ``i = sigmoid(W_i [x, h] + b_i + p_i * c)``, ``f = sigmoid(W_f [x, h]
    + b_f + forget_bias + p_f * c)``, ``c' = f * c + i * tanh(W_j [x, h] + b_j)``,
    ``o = sigmoid(W_o [x, h] + b_o + p_o * c')``, ``h' = o * tanh(c')``. The
    kernel and the peephole weights are Glorot-uniform (TensorFlow's default
    variable initialiser, on the kernel shape ``[input + hidden, 4 hidden]``
    and on each ``[hidden]`` peephole vector) and the bias is zero.

    Parameters
    ----------
    input_size : int
        Features per step.
    hidden_size : int
        Units.
    forget_bias : float
        Constant added to the forget gate.
    """

    def __init__(self, input_size: int, hidden_size: int, forget_bias: float):
        super().__init__()
        self.hidden_size, self.forget_bias = hidden_size, forget_bias
        self.input_weight = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.hidden_weight = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.peephole = nn.Parameter(torch.empty(3, hidden_size))
        kernel_limit = math.sqrt(6.0 / (input_size + hidden_size + 4 * hidden_size))
        nn.init.uniform_(self.input_weight, -kernel_limit, kernel_limit)
        nn.init.uniform_(self.hidden_weight, -kernel_limit, kernel_limit)
        peephole_limit = math.sqrt(6.0 / (2 * hidden_size))
        nn.init.uniform_(self.peephole, -peephole_limit, peephole_limit)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the layer over a sequence.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, input_size]``.
        state : tuple of torch.Tensor
            Initial ``(h, c)``, each ``[B, hidden_size]``.

        Returns
        -------
        output : torch.Tensor
            ``[B, T, hidden_size]``.
        state : tuple of torch.Tensor
            Final ``(h, c)``.
        """
        hidden, cell = state
        projected = F.linear(x, self.input_weight, self.bias)
        outputs = []
        for step in range(x.shape[1]):
            gates = projected[:, step] + F.linear(hidden, self.hidden_weight)
            gate_input, candidate, gate_forget, gate_output = gates.chunk(4, dim=1)
            gate_input = torch.sigmoid(gate_input + self.peephole[0] * cell)
            gate_forget = torch.sigmoid(gate_forget + self.forget_bias + self.peephole[1] * cell)
            cell = gate_forget * cell + gate_input * torch.tanh(candidate)
            gate_output = torch.sigmoid(gate_output + self.peephole[2] * cell)
            hidden = gate_output * torch.tanh(cell)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, cell)


class DeepSleepNet(nn.Module):
    """DeepSleepNet: DeepFeatureNet, then sequence residual learning (``model.py:224-425``).

    Each epoch of a ``[B, T, C, L]`` run is encoded by the CNNs; the features
    go through a fully connected shortcut (no bias, batch norm, ReLU) and
    through two independent stacks of peephole LSTMs, one reading the run
    forward and one backward, each layer followed by dropout. The two
    directions are concatenated, the shortcut is added, dropout is applied
    and a linear layer gives one logit row per epoch. With a state, the run
    continues a night (upstream feeds each direction's final state of one
    step as the next step's initial state); without, both start at zero.

    Parameters
    ----------
    features : DeepFeatureNet
        Epoch encoder.
    hidden_size : int
        LSTM units per direction.
    n_layers : int
        LSTM layers per direction.
    n_classes : int
        Stages.
    dropout : float
        Dropout probability of the sequence part.
    forget_bias : float
        LSTM forget-gate constant.
    norm : dict
        Batch-normalisation settings of the shortcut.
    """

    def __init__(self, features: DeepFeatureNet, hidden_size: int, n_layers: int, n_classes: int,
                 dropout: float, forget_bias: float, norm: dict):
        super().__init__()
        self.features = features
        size = features.feature_size
        self.hidden_size, self.n_layers = hidden_size, n_layers
        linear = nn.Linear(size, 2 * hidden_size, bias=False)
        he_truncated_normal_(linear.weight, size)
        self.shortcut = nn.Sequential(linear, batch_norm(2 * hidden_size, norm), nn.ReLU())
        self.forward_layers = nn.ModuleList(
            [PeepholeLSTM(size if layer == 0 else hidden_size, hidden_size, forget_bias)
             for layer in range(n_layers)])
        self.backward_layers = nn.ModuleList(
            [PeepholeLSTM(size if layer == 0 else hidden_size, hidden_size, forget_bias)
             for layer in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, n_classes)
        he_truncated_normal_(self.classifier.weight, 2 * hidden_size)
        nn.init.zeros_(self.classifier.bias)

    def zero_state(self, batch: int, device: torch.device) -> State:
        """Zero initial state of both directions.

        Parameters
        ----------
        batch : int
            Sequences in the batch.
        device : torch.device
            Device of the state.

        Returns
        -------
        State
            Per direction, one ``(h, c)`` pair per layer.
        """
        def zeros() -> list[tuple[torch.Tensor, torch.Tensor]]:
            """One direction's zero state.

            Returns
            -------
            list
                ``(h, c)`` per layer.
            """
            return [(torch.zeros(batch, self.hidden_size, device=device),
                     torch.zeros(batch, self.hidden_size, device=device))
                    for _ in range(self.n_layers)]

        return zeros(), zeros()

    def run_stack(self, layers: nn.ModuleList, x: torch.Tensor,
                  state: list[tuple[torch.Tensor, torch.Tensor]]
                  ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One direction: the LSTM layers, each output followed by dropout.

        Parameters
        ----------
        layers : nn.ModuleList
            The direction's layers.
        x : torch.Tensor
            ``[B, T, feature_size]`` in the direction's reading order.
        state : list
            Initial ``(h, c)`` per layer.

        Returns
        -------
        output : torch.Tensor
            ``[B, T, hidden_size]``.
        state : list
            Final ``(h, c)`` per layer.
        """
        final = []
        for layer, initial in zip(layers, state):
            x, last = layer(x, initial)
            x = self.dropout(x)
            final.append(last)
        return x, final

    def step(self, x: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        """Logits of every epoch of a run, continuing from a state.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, C, L]``.
        state : State
            Initial state (:meth:`zero_state`).

        Returns
        -------
        logits : torch.Tensor
            ``[B, T, n_classes]``.
        state : State
            Final state of each direction.
        """
        batch, steps = x.shape[:2]
        features = self.features(x.flatten(0, 1))
        shortcut = self.shortcut(features)
        sequence = features.view(batch, steps, -1)
        forward_output, forward_state = self.run_stack(self.forward_layers, sequence, state[0])
        backward_output, backward_state = self.run_stack(self.backward_layers,
                                                         sequence.flip(1), state[1])
        output = torch.cat([forward_output, backward_output.flip(1)], dim=-1).flatten(0, 1)
        logits = self.classifier(self.dropout(output + shortcut))
        return logits.view(batch, steps, -1), (forward_state, backward_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of every epoch of independent runs, each starting from a zero state.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, C, L]``.

        Returns
        -------
        torch.Tensor
            ``[B, T, n_classes]``.
        """
        return self.step(x, self.zero_state(x.shape[0], x.device))[0]


def build_features(spec: dict, in_channels: int, n_samples: int,
                   sampling_rate: float) -> DeepFeatureNet:
    """DeepFeatureNet from a configuration's ``model`` section.

    Parameters
    ----------
    spec : dict
        ``model`` section of ``native.yaml`` or ``ours.yaml``.
    in_channels : int
        Input channels.
    n_samples : int
        Samples per epoch.
    sampling_rate : float
        Hz.

    Returns
    -------
    DeepFeatureNet
        The epoch encoder.
    """
    return DeepFeatureNet(in_channels, n_samples, sampling_rate, spec["first_filters"],
                          spec["deep_filters"], spec["small"], spec["large"], spec["dropout"],
                          spec["batch_norm"])


def build_network(spec: dict, features: DeepFeatureNet, n_classes: int) -> DeepSleepNet:
    """DeepSleepNet around an epoch encoder, from a configuration's ``model`` section.

    Parameters
    ----------
    spec : dict
        ``model`` section.
    features : DeepFeatureNet
        Epoch encoder (pretrained or fresh).
    n_classes : int
        Stages.

    Returns
    -------
    DeepSleepNet
        The network.
    """
    return DeepSleepNet(features, spec["hidden_size"], spec["n_layers"], n_classes,
                        spec["dropout"], spec["forget_bias"], spec["batch_norm"])
