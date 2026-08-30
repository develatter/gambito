"""Self-play JSONL -> training samples: the data half of the AlphaZero loop.

The Rust generator (gambito-ai/examples/selfplay.rs) writes one JSON line per
position: {"fen": ..., "pi": "e2e4:40 g1f3:12 ...", "z": 1.0}. Unlike the
supervised bootstrap, the policy target is not a single move but the full
root visit distribution of the search — the net learns to predict where a
deeper search will spend its effort.
"""

import json
from pathlib import Path

import chess
import numpy as np

# The generator's --sims bounds how many root moves can carry visits, so a
# fixed-size padded target is safe as long as MAX_PI >= sims.
MAX_PI = 128


def build_selfplay_dataset(jsonl_path: str | Path, out_path: str | Path) -> int:
    """Packs a self-play JSONL into a compact .npz; returns the sample count."""
    fens: list[str] = []
    pis: list[str] = []
    zs: list[int] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fens.append(d["fen"])
            pis.append(d["pi"])
            zs.append(int(d["z"]))
    np.savez_compressed(
        out_path,
        fens=np.array(fens, dtype="S"),
        pis=np.array(pis, dtype="S"),
        z=np.array(zs, dtype=np.int8),
    )
    return len(fens)


class SelfplayDataset:
    """Map-style torch dataset over a .npz built by `build_selfplay_dataset`.

    Yields (planes, idx[MAX_PI], prob[MAX_PI], z): a sparse padded policy
    distribution. Padding rows have prob 0, so they vanish in the soft
    cross-entropy regardless of their index.
    """

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=False)
        self.fens = data["fens"]
        self.pis = data["pis"]
        self.z = data["z"]

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, i: int):
        import torch

        from .encoding import encode_planes, policy_index

        board = chess.Board(self.fens[i].decode())
        planes = torch.from_numpy(encode_planes(board))
        idx = torch.zeros(MAX_PI, dtype=torch.long)
        prob = torch.zeros(MAX_PI)
        entries = self.pis[i].decode().split()[:MAX_PI]
        visits = [int(e.split(":")[1]) for e in entries]
        total = sum(visits)
        for k, (entry, v) in enumerate(zip(entries, visits)):
            move = chess.Move.from_uci(entry.split(":")[0])
            idx[k] = policy_index(move, board.turn)
            prob[k] = v / total
        return planes, idx, prob, float(self.z[i])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pack self-play JSONL into a training .npz")
    ap.add_argument("jsonl", help="output of the Rust selfplay generator")
    ap.add_argument("out", help="output .npz path")
    args = ap.parse_args()
    n = build_selfplay_dataset(args.jsonl, args.out)
    print(f"{n:,} samples -> {args.out}")


if __name__ == "__main__":
    main()
