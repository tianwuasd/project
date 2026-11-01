"""Stacked bidirectional ConvLSTM; center-frame supervision for seven-slice windows."""

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(inputs + hidden, 4 * hidden, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


class BiConvLSTM(nn.Module):
    def __init__(self, channels=32, layers=2, classes=3):
        super().__init__()
        self.channels = channels
        self.forward_cells = nn.ModuleList(
            [ConvLSTMCell(channels if i == 0 else 2 * channels, channels) for i in range(layers)]
        )
        self.backward_cells = nn.ModuleList(
            [ConvLSTMCell(channels if i == 0 else 2 * channels, channels) for i in range(layers)]
        )
        self.head = nn.Conv2d(2 * channels, classes, 1)

    def forward(self, sequence):
        if sequence.ndim != 5 or sequence.shape[1] % 2 != 1:
            raise ValueError("Expected [batch, odd time, channels, height, width]")
        b, t, _, h, w = sequence.shape
        for fw, bw in zip(self.forward_cells, self.backward_cells):
            state_f = (sequence.new_zeros(b, self.channels, h, w), sequence.new_zeros(b, self.channels, h, w))
            state_b = (torch.zeros_like(state_f[0]), torch.zeros_like(state_f[1]))
            forward, backward = [], []
            for i in range(t):
                state_f = fw(sequence[:, i], state_f)
                state_b = bw(sequence[:, t - 1 - i], state_b)
                forward.append(state_f[0])
                backward.append(state_b[0])
            sequence = torch.stack([torch.cat([a, b], 1) for a, b in zip(forward, reversed(backward))], dim=1)
        return self.head(sequence[:, t // 2])
