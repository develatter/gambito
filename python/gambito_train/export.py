"""Checkpoint -> model.cyw exporter (int8, weights-only quantization).

The .cyw format is defined in docs/encoding.md. Two transformations happen
here, both chosen to keep the Rust inference trivial:

* BatchNorm folding: every Conv+BN pair collapses into a single conv with
  bias, so Rust never sees a BatchNorm.
* Symmetric per-layer int8: q = round(w / scale), scale = max|w| / 127.
  Biases stay f32. Rust dequantizes at load time and runs plain f32 math.

Also dumps a golden file (fen -> value + priors) computed with the SAME
dequantized weights, so the Rust forward pass can be verified exactly.

Usage (from python/):
  uv run python -m gambito_train.export checkpoints/epoch3.pt model.cyw --golden nn_golden.txt
"""

import argparse
import struct

import chess
import numpy as np
import torch

from .encoding import encode_planes, policy_index
from .model import GambitoNet, NetConfig

MAGIC = b"CYW1"

GOLDEN_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2",
    "4k3/8/4K3/8/8/8/8/R7 w - - 0 1",
    "5k2/8/8/8/8/8/6p1/5K2 b - - 0 1",
]


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """Collapses conv (bias-free) + eval-mode BN into (weight, bias)."""
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight / std
    w = conv.weight * gamma.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * gamma
    return w.detach(), b.detach()


def quantize(w: torch.Tensor):
    """Symmetric per-layer int8; returns (q, scale, dequantized f32)."""
    scale = float(w.abs().max().clamp(min=1e-12)) / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale, q.float() * scale


def collect_layers(model: GambitoNet):
    """(name, weight, bias) in the exact order Rust reads them."""
    layers = [("stem", *fold_conv_bn(model.stem[0], model.stem[1]))]
    for i, block in enumerate(model.blocks):
        layers.append((f"block{i}.conv1", *fold_conv_bn(block.conv1, block.bn1)))
        layers.append((f"block{i}.conv2", *fold_conv_bn(block.conv2, block.bn2)))
    layers.append(("policy_conv", *fold_conv_bn(model.policy_conv[0], model.policy_conv[1])))
    layers.append(("policy_fc", model.policy_fc.weight.detach(), model.policy_fc.bias.detach()))
    layers.append(("value_conv", *fold_conv_bn(model.value_conv[0], model.value_conv[1])))
    layers.append(("value_fc1", model.value_fc[0].weight.detach(), model.value_fc[0].bias.detach()))
    layers.append(("value_fc2", model.value_fc[2].weight.detach(), model.value_fc[2].bias.detach()))
    return layers


def export(checkpoint_path: str, out_path: str, golden_path: str | None) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = NetConfig(**ckpt["config"])
    model = GambitoNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dequant: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    blob = bytearray(MAGIC)
    blob += struct.pack(
        "<5I", cfg.channels, cfg.blocks, cfg.policy_channels, cfg.value_channels, cfg.value_hidden
    )
    for name, w, b in collect_layers(model):
        q, scale, dq = quantize(w)
        dequant[name] = (dq, b)
        blob += q.numpy().tobytes()
        blob += struct.pack("<f", scale)
        blob += b.numpy().astype(np.float32).tobytes()

    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"{out_path}: {len(blob):,} bytes")

    if golden_path:
        dump_golden(cfg, dequant, golden_path)


def _conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, relu: bool = True):
    pad = 1 if w.shape[-1] == 3 else 0
    y = torch.nn.functional.conv2d(x, w, b, padding=pad)
    return torch.relu(y) if relu else y


def forward_dequant(cfg: NetConfig, L: dict, x: torch.Tensor):
    """The exact function Rust implements: folded convs, dequantized f32."""
    x = _conv(x, *L["stem"])
    for i in range(cfg.blocks):
        y = _conv(x, *L[f"block{i}.conv1"])
        y = _conv(y, *L[f"block{i}.conv2"], relu=False)
        x = torch.relu(x + y)
    p = _conv(x, *L["policy_conv"]).flatten(1)
    policy = p @ L["policy_fc"][0].T + L["policy_fc"][1]
    v = _conv(x, *L["value_conv"]).flatten(1)
    v = torch.relu(v @ L["value_fc1"][0].T + L["value_fc1"][1])
    value = torch.tanh(v @ L["value_fc2"][0].T + L["value_fc2"][1])
    return policy[0], value[0, 0]


def dump_golden(cfg: NetConfig, L: dict, path: str) -> None:
    """Lines of: fen | value | uci:prior uci:prior ..."""
    lines = []
    with torch.no_grad():
        for fen in GOLDEN_FENS:
            board = chess.Board(fen)
            x = torch.from_numpy(encode_planes(board)).unsqueeze(0)
            logits, value = forward_dequant(cfg, L, x)
            legal = {policy_index(m, board.turn): m.uci() for m in board.legal_moves}
            idx = sorted(legal)
            sub = logits[idx]
            priors = torch.softmax(sub, dim=0)
            pairs = " ".join(f"{legal[i]}:{p.item():.6f}" for i, p in zip(idx, priors))
            lines.append(f"{fen} | {value.item():.6f} | {pairs}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} positions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("out", help="output .cyw path")
    ap.add_argument("--golden", default=None, help="also dump a golden eval file")
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.golden)


if __name__ == "__main__":
    main()
