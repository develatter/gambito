"""The evaluator network: a small AlphaZero-style residual convnet.

Input  [B, 19, 8, 8]  (see encoding.py / docs/encoding.md)
Output policy logits [B, 4168], value [B] in [-1, 1]

Sized for the project's budget: ~1.5M parameters, so the int8 export stays
around 1.5 MB and embeds comfortably in the gambito binary. MCTS in front
amplifies whatever strength the net has; small is fine.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PLANE_COUNT, POLICY_SIZE


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 5
    policy_channels: int = 4
    value_channels: int = 8
    value_hidden: int = 64


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = relu(x + f(x)).

    The skip lets gradients flow straight through, which is what makes a
    10-conv-deep net trainable; f only has to learn the *correction*.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GambitoNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        c = cfg.channels

        self.stem = nn.Sequential(
            nn.Conv2d(PLANE_COUNT, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(c) for _ in range(cfg.blocks)])

        # Policy head: squeeze channels with a 1x1 conv, then one dense layer
        # to the 4168 move slots. The dense layer dominates the param count,
        # hence the aggressive squeeze.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, cfg.policy_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.policy_channels),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(cfg.policy_channels * 64, POLICY_SIZE)

        # Value head: squeeze, dense to a small hidden layer, then a single
        # tanh scalar — the same [-1, 1] the MCTS backs up.
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, cfg.value_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.value_channels),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(cfg.value_channels * 64, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.blocks(self.stem(x))
        policy = self.policy_fc(self.policy_conv(trunk).flatten(1))
        value = torch.tanh(self.value_fc(self.value_conv(trunk).flatten(1))).squeeze(-1)
        return policy, value


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
