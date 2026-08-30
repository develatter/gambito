"""Pure-Python spec tests, the mirror of the Rust unit tests in encode.rs.
Both sides must pass these AND the cross-language golden test."""

import chess
import numpy as np

from gambito_train.encoding import encode_planes, policy_index

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_startpos_looks_identical_from_both_sides():
    white = encode_planes(chess.Board(f"{STARTPOS} w KQkq - 0 1"))
    black = encode_planes(chess.Board(f"{STARTPOS} b KQkq - 0 1"))
    assert np.array_equal(white, black)


def test_startpos_plane_shapes():
    t = encode_planes(chess.Board())
    assert t[0].sum() == 8.0 and t[0, 1].all()  # our pawns on rank 2
    assert t[12:16].all()  # all castling rights
    assert t[16].sum() == 0.0  # no en passant
    assert (t[17] == 0.0).all() and t[18].all()


def test_en_passant_square_is_mirrored_for_black():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2")
    t = encode_planes(board)
    assert t[16].sum() == 1.0 and t[16, 5, 5] == 1.0  # f3 mirrors to f6


def test_policy_index_spec():
    w, b = chess.WHITE, chess.BLACK
    assert policy_index(chess.Move.from_uci("e2e4"), w) == 796
    assert policy_index(chess.Move.from_uci("e7e5"), b) == 796  # mirrored twin
    assert policy_index(chess.Move.from_uci("a7a8q"), w) == 3128
    assert policy_index(chess.Move.from_uci("a7a8n"), w) == 4096
    assert policy_index(chess.Move.from_uci("a7b8r"), w) == 4104
    assert policy_index(chess.Move.from_uci("g2g1b"), b) == 4151
