//! Pseudo-legal generation plus a legality filter (copy-make makes the
//! filter cheap enough for M1; MCTS in M2 wants fully legal lists anyway).

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveFlags, MoveList};
use crate::position::Position;
use crate::types::{Color, PieceKind, Square};

pub fn legal_moves(pos: &Position) -> MoveList {
    let mut legal = MoveList::new();
    let us = pos.side_to_move;
    for &mv in &pseudo_legal(pos) {
        if !pos.apply(mv).in_check(us) {
            legal.push(mv);
        }
    }
    legal
}

fn pseudo_legal(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    let us = pos.side_to_move;
    let them = us.opposite();
    let occupied = pos.occupied();
    let enemies = pos.occupied_by(them);
    let empty = !occupied;

    pawn_moves(pos, us, &mut list);

    for from in pos.pieces(us, PieceKind::Knight) {
        push_targets(&mut list, from, attacks::knight_attacks(from), enemies, empty);
    }
    for from in pos.pieces(us, PieceKind::Bishop) {
        push_targets(&mut list, from, attacks::bishop_attacks(from, occupied), enemies, empty);
    }
    for from in pos.pieces(us, PieceKind::Rook) {
        push_targets(&mut list, from, attacks::rook_attacks(from, occupied), enemies, empty);
    }
    for from in pos.pieces(us, PieceKind::Queen) {
        push_targets(&mut list, from, attacks::queen_attacks(from, occupied), enemies, empty);
    }

    let king = pos.king_square(us);
    push_targets(&mut list, king, attacks::king_attacks(king), enemies, empty);
    castling_moves(pos, us, king, &mut list);

    list
}

#[inline]
fn push_targets(list: &mut MoveList, from: Square, targets: Bitboard, enemies: Bitboard, empty: Bitboard) {
    for to in targets & enemies {
        list.push(Move::new(from, to, MoveFlags::Capture));
    }
    for to in targets & empty {
        list.push(Move::new(from, to, MoveFlags::Quiet));
    }
}

fn pawn_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let them = us.opposite();
    let occupied = pos.occupied();
    let enemies = pos.occupied_by(them);
    let (dir, start_rank, promo_rank) = match us {
        Color::White => (1i8, 1u8, 7u8),
        Color::Black => (-1i8, 6u8, 0u8),
    };

    for from in pos.pieces(us, PieceKind::Pawn) {
        // Pushes.
        if let Some(one) = from.offset(0, dir) {
            if !occupied.contains(one) {
                if one.rank() == promo_rank {
                    push_promotions(list, from, one, false);
                } else {
                    list.push(Move::new(from, one, MoveFlags::Quiet));
                    if from.rank() == start_rank {
                        let two = from.offset(0, 2 * dir).expect("double push stays on board");
                        if !occupied.contains(two) {
                            list.push(Move::new(from, two, MoveFlags::DoublePush));
                        }
                    }
                }
            }
        }
        // Captures, including en passant.
        for to in attacks::pawn_attacks(us, from) {
            if enemies.contains(to) {
                if to.rank() == promo_rank {
                    push_promotions(list, from, to, true);
                } else {
                    list.push(Move::new(from, to, MoveFlags::Capture));
                }
            } else if pos.en_passant == Some(to) {
                list.push(Move::new(from, to, MoveFlags::EnPassant));
            }
        }
    }
}

fn push_promotions(list: &mut MoveList, from: Square, to: Square, capture: bool) {
    for kind in [PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight] {
        list.push(Move::new(from, to, MoveFlags::promo(kind, capture)));
    }
}

fn castling_moves(pos: &Position, us: Color, king: Square, list: &mut MoveList) {
    if pos.in_check(us) {
        return;
    }
    let them = us.opposite();
    let occupied = pos.occupied();
    let rank = if us == Color::White { 0 } else { 7 };
    // The king must not pass through or land on an attacked square; the
    // rook path only needs to be empty (b-file square matters for O-O-O).
    if pos.castling.king_side(us) {
        let f = Square::new(5, rank);
        let g = Square::new(6, rank);
        if !occupied.contains(f)
            && !occupied.contains(g)
            && !pos.is_attacked(f, them)
            && !pos.is_attacked(g, them)
        {
            list.push(Move::new(king, g, MoveFlags::CastleKing));
        }
    }
    if pos.castling.queen_side(us) {
        let b = Square::new(1, rank);
        let c = Square::new(2, rank);
        let d = Square::new(3, rank);
        if !occupied.contains(b)
            && !occupied.contains(c)
            && !occupied.contains(d)
            && !pos.is_attacked(c, them)
            && !pos.is_attacked(d, them)
        {
            list.push(Move::new(king, c, MoveFlags::CastleQueen));
        }
    }
}
