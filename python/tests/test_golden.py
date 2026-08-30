"""Cross-language contract test: the tensors Rust produced (dump_golden)
must match this package's encoder exactly. If this fails, one side drifted
from docs/encoding.md — fix the drift, never the golden file by hand.

Regenerate the golden file from the repo root with:
  cargo run -p gambito-ai --example dump_golden > python/tests/golden/positions.json
"""

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from gambito_train.encoding import PLANE_COUNT, encode_planes, policy_index

GOLDEN = Path(__file__).parent / "golden" / "positions.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists(), reason="golden file missing: run dump_golden first"
)


def entries():
    return json.loads(GOLDEN.read_text())


def test_planes_match_rust():
    for entry in entries():
        board = chess.Board(entry["fen"])
        rust = np.array(entry["planes"], dtype=np.float32).reshape(PLANE_COUNT, 8, 8)
        ours = encode_planes(board)
        np.testing.assert_allclose(ours, rust, atol=1e-6, err_msg=entry["fen"])


def test_policy_indices_match_rust():
    for entry in entries():
        board = chess.Board(entry["fen"])
        rust = entry["moves"]
        ours = {m.uci(): policy_index(m, board.turn) for m in board.legal_moves}
        assert ours == rust, f"policy drift on {entry['fen']}"
