"""Supervised bootstrap trainer.

Usage (from python/):
  uv run python -m gambito_train.dataset lichess.pgn.zst data.npz --max-games 50000
  uv run python -m gambito_train.train data.npz --epochs 2

Each checkpoint stores the model state plus its NetConfig, so inference and
the future .cyw export never have to guess dimensions.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .dataset import SupervisedDataset
from .model import GambitoNet, NetConfig, parameter_count


def loss_fn(
    policy_logits: torch.Tensor,  # [B, 4168]
    value: torch.Tensor,  # [B], tanh output in [-1, 1]
    idx: torch.Tensor,  # [B, K] policy indices of the target distribution
    prob: torch.Tensor,  # [B, K] their probabilities (rows sum to 1; pad = 0)
    target_z: torch.Tensor,  # [B], game outcome in {-1, 0, 1}
) -> torch.Tensor:
    # AlphaZero's L = (z - v)^2 - pi^T log(p), with the value term at 0.5:
    # z is the noisiest label in both modes (flagged wins in human games,
    # temperature blunders in self-play) — let policy lead. pi is sparse:
    # supervised rows are one-hot (K=1), self-play rows are MCTS visit
    # distributions; padding entries have prob 0 and drop out of the sum.
    logp = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(prob * logp.gather(1, idx)).sum(dim=1).mean()
    return policy_loss + 0.5 * F.mse_loss(value, target_z)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total = correct = 0
    loss_sum = value_err = 0.0
    for planes, idx, prob, target_z in loader:
        planes = planes.to(device)
        idx = idx.to(device)
        prob = prob.to(device)
        target_z = target_z.float().to(device)
        policy, value = model(planes)
        loss_sum += loss_fn(policy, value, idx, prob, target_z).item() * len(planes)
        # "Accuracy" = did the net's top move match the target's top move.
        top_target = idx.gather(1, prob.argmax(dim=1, keepdim=True)).squeeze(1)
        correct += (policy.argmax(dim=1) == top_target).sum().item()
        value_err += (value - target_z).abs().sum().item()
        total += len(planes)
    return {
        "loss": loss_sum / total,
        "move_acc": correct / total,
        "value_mae": value_err / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+", help=".npz file(s); mixing supervised and self-play is fine")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    args = ap.parse_args()

    import numpy as np
    from torch.utils.data import ConcatDataset

    from .selfplay import SelfplayDataset

    # Self-play .npz files carry visit distributions ("pis"); supervised
    # ones carry single played moves ("moves"). Same loss and target shapes
    # either way, so a replay-buffer mix is just a concatenation.
    parts = []
    for path in args.data:
        with np.load(path, allow_pickle=False) as probe:
            selfplay = "pis" in probe.files
        parts.append(SelfplayDataset(path) if selfplay else SupervisedDataset(path))
    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    val_len = max(1, len(dataset) // 50)
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_len, val_len],
        generator=torch.Generator().manual_seed(7),
    )
    # Python 3.14 defaults multiprocessing to forkserver, which PICKLES the
    # whole dataset into every worker (25M samples x N workers = OOM). fork
    # shares it copy-on-write instead; workers never touch CUDA, so it's safe.
    ctx = "fork" if args.workers > 0 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        multiprocessing_context=ctx,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=2, multiprocessing_context="fork"
    )

    cfg = NetConfig()
    model = GambitoNet(cfg).to(args.device)
    if args.init:
        ckpt = torch.load(args.init, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"fine-tuning from {args.init} (val {ckpt.get('val')})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"{parameter_count(model):,} params · {len(train_set):,} train / "
        f"{len(val_set):,} val samples · device {args.device}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for step, (planes, idx, prob, target_z) in enumerate(train_loader, 1):
            planes = planes.to(args.device, non_blocking=True)
            idx = idx.to(args.device, non_blocking=True)
            prob = prob.to(args.device, non_blocking=True)
            target_z = target_z.float().to(args.device, non_blocking=True)
            loss = loss_fn(*model(planes), idx, prob, target_z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0:
                rate = step * args.batch_size / (time.time() - start)
                print(f"epoch {epoch} step {step:5d} loss {loss.item():.4f} ({rate:,.0f} pos/s)")

        metrics = evaluate(model, val_loader, args.device)
        print(
            f"epoch {epoch} val: loss {metrics['loss']:.4f} "
            f"move_acc {metrics['move_acc']:.1%} value_mae {metrics['value_mae']:.3f}"
        )
        torch.save(
            {"model": model.state_dict(), "config": asdict(cfg), "val": metrics},
            out_dir / f"epoch{epoch}.pt",
        )


if __name__ == "__main__":
    main()
