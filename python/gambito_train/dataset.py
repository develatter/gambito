"""PGN -> training samples for the supervised bootstrap.

Two stages, so the expensive PGN replay happens once:

1. `build_dataset` streams a Lichess PGN (.pgn or .pgn.zst), filters games,
   and writes a compact .npz of (fen, played move, outcome) triples —
   ~100 bytes per sample instead of the 4.8 KB of an encoded tensor.
2. `SupervisedDataset` re-encodes positions to planes on the fly inside
   DataLoader workers, trading a little CPU for 50x less disk.

Targets per docs/encoding.md: the policy target is the index of the move a
strong human actually played; the value target is the game outcome z from
the side to move's perspective.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import numpy as np

RESULT_TO_Z = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _as_bytes(a: np.ndarray) -> np.ndarray:
    """Backward compat: older .npz files stored unicode arrays."""
    return a if a.dtype.kind == "S" else np.char.encode(a, "ascii")


def open_pgn(path: str | Path):
    """Opens a .pgn or Lichess .pgn.zst as a text stream."""
    path = Path(path)
    if path.suffix == ".zst":
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(raw, encoding="utf-8")
    return open(path, encoding="utf-8")


def _elo(headers: chess.pgn.Headers, key: str) -> int:
    try:
        return int(headers.get(key, 0))
    except ValueError:
        return 0


def _base_seconds(headers: chess.pgn.Headers) -> int:
    tc = headers.get("TimeControl", "-")
    if tc == "-":  # correspondence: unlimited thinking time, keep it
        return 10**9
    try:
        return int(tc.split("+")[0])
    except ValueError:
        return 0


def keep_game(headers: chess.pgn.Headers, min_elo: int, min_base_seconds: int) -> bool:
    return (
        headers.get("Result") in RESULT_TO_Z
        and headers.get("Variant", "Standard") == "Standard"
        and "FEN" not in headers
        and _elo(headers, "WhiteElo") >= min_elo
        and _elo(headers, "BlackElo") >= min_elo
        and _base_seconds(headers) >= min_base_seconds
    )


def iter_samples(
    stream,
    min_elo: int = 1800,
    min_base_seconds: int = 180,
    max_games: int | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Yields (fen, uci move, z from side to move's POV) over filtered games."""
    kept = 0
    while max_games is None or kept < max_games:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        if not keep_game(game.headers, min_elo, min_base_seconds):
            continue
        kept += 1
        z_white = RESULT_TO_Z[game.headers["Result"]]
        board = game.board()
        for move in game.mainline_moves():
            z = z_white if board.turn == chess.WHITE else -z_white
            yield board.fen(), move.uci(), z
            board.push(move)


def build_dataset(
    pgn_path: str | Path,
    out_path: str | Path,
    min_elo: int = 1800,
    min_base_seconds: int = 180,
    max_games: int | None = None,
) -> int:
    """Streams a PGN into a compact .npz; returns the sample count."""
    fens: list[str] = []
    moves: list[str] = []
    zs: list[int] = []
    with open_pgn(pgn_path) as stream:
        for fen, uci, z in iter_samples(stream, min_elo, min_base_seconds, max_games):
            fens.append(fen)
            moves.append(uci)
            zs.append(z)
    # dtype "S" (bytes) is 4x smaller in RAM than numpy's default 4-bytes-
    # per-char unicode — at 25M samples that is the difference between a
    # 2 GB and a 9 GB resident dataset.
    np.savez_compressed(
        out_path,
        fens=np.array(fens, dtype="S"),
        moves=np.array(moves, dtype="S"),
        z=np.array(zs, dtype=np.int8),
    )
    return len(fens)


class SupervisedDataset:
    """Map-style torch dataset over a .npz built by `build_dataset`.

    Encoding happens here, in __getitem__, so DataLoader workers parallelize
    it. Imports torch lazily to keep the PGN tooling usable without it.
    """

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=False)
        self.fens = _as_bytes(data["fens"])
        self.moves = _as_bytes(data["moves"])
        self.z = data["z"]

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, i: int):
        import torch

        from .encoding import encode_planes, policy_index

        board = chess.Board(self.fens[i].decode())
        move = chess.Move.from_uci(self.moves[i].decode())
        planes = torch.from_numpy(encode_planes(board))
        # One-hot soft target: the same (idx, prob) shape SelfplayDataset
        # yields, so one loss function serves both training modes.
        idx = torch.tensor([policy_index(move, board.turn)], dtype=torch.long)
        prob = torch.ones(1)
        return planes, idx, prob, float(self.z[i])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build a training .npz from a PGN")
    ap.add_argument("pgn", help=".pgn or .pgn.zst file")
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--min-elo", type=int, default=1800)
    ap.add_argument("--min-base-seconds", type=int, default=180)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    n = build_dataset(args.pgn, args.out, args.min_elo, args.min_base_seconds, args.max_games)
    print(f"{n:,} samples -> {args.out}")


if __name__ == "__main__":
    main()
