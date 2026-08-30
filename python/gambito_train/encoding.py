"""Position and move encoding per docs/encoding.md — the Python half of the
Rust <-> Python tensor contract. Must match crates/gambito-ai/src/encode.rs
exactly; the golden tests in tests/test_golden.py enforce it."""

import chess
import numpy as np

PLANE_COUNT = 19
POLICY_SIZE = 4168

# python-chess piece constants in PieceKind::index() order: P N B R Q K.
_KIND_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def pov_square(sq: int, turn: chess.Color) -> int:
    """Mirror ranks (not files) when Black is to move, so "us" plays up."""
    return sq if turn == chess.WHITE else sq ^ 56


def encode_planes(board: chess.Board) -> np.ndarray:
    """The [19, 8, 8] f32 tensor from docs/encoding.md, us-POV frame."""
    t = np.zeros((PLANE_COUNT, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us

    for offset, color in ((0, us), (6, them)):
        for i, kind in enumerate(_KIND_ORDER):
            for sq in board.pieces(kind, color):
                p = pov_square(sq, us)
                t[offset + i, p // 8, p % 8] = 1.0

    rights = (
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    )
    for i, on in enumerate(rights):
        if on:
            t[12 + i] = 1.0

    if board.ep_square is not None:
        p = pov_square(board.ep_square, us)
        t[16, p // 8, p % 8] = 1.0

    t[17] = board.halfmove_clock / 100.0
    t[18] = 1.0
    return t


def policy_index(move: chess.Move, turn: chess.Color) -> int:
    """Policy-head index (0..4168): from*64+to for normal moves and queen
    promotions; 4096 + from_file*9 + direction*3 + piece for N/B/R
    underpromotions (direction 0 = push, 1 = toward file-1, 2 = file+1)."""
    frm = pov_square(move.from_square, turn)
    to = pov_square(move.to_square, turn)
    if move.promotion is None or move.promotion == chess.QUEEN:
        return frm * 64 + to
    if to % 8 == frm % 8:
        direction = 0
    elif to % 8 == frm % 8 - 1:
        direction = 1
    else:
        direction = 2
    return 4096 + (frm % 8) * 9 + direction * 3 + _UNDERPROMO[move.promotion]
