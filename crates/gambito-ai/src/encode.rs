//! Position and move encoding per docs/encoding.md — the Rust half of the
//! Rust <-> Python tensor contract. Golden tests on both sides compare this
//! against python/gambito_train/encoding.py byte for byte.

use gambito_engine::{Color, Move, PieceKind, Position, Square};

pub const PLANE_COUNT: usize = 19;
pub const POLICY_SIZE: usize = 4168;

/// Maps a board square into the side-to-move frame: for Black, ranks are
/// mirrored (files untouched) so "us" always plays up the board.
#[inline]
pub fn pov_square(sq: Square, side: Color) -> Square {
    match side {
        Color::White => sq,
        Color::Black => Square(sq.0 ^ 56),
    }
}

/// Encodes `pos` as the `[19, 8, 8]` f32 tensor from docs/encoding.md,
/// flattened as `plane * 64 + rank * 8 + file` in the us-POV frame.
pub fn encode_planes(pos: &Position) -> Vec<f32> {
    let mut t = vec![0.0f32; PLANE_COUNT * 64];
    let us = pos.side_to_move;
    let them = us.opposite();

    // Planes 0-5 us pieces, 6-11 them, in PieceKind order.
    for (offset, color) in [(0, us), (6, them)] {
        for kind in PieceKind::ALL {
            let plane = offset + kind.index();
            for sq in pos.pieces(color, kind) {
                t[plane * 64 + pov_square(sq, us).index()] = 1.0;
            }
        }
    }

    // Planes 12-15: castling rights as constant 0/1 planes.
    let rights = [
        pos.castling.king_side(us),
        pos.castling.queen_side(us),
        pos.castling.king_side(them),
        pos.castling.queen_side(them),
    ];
    for (i, on) in rights.into_iter().enumerate() {
        if on {
            t[(12 + i) * 64..(13 + i) * 64].fill(1.0);
        }
    }

    // Plane 16: en-passant target one-hot.
    if let Some(ep) = pos.en_passant {
        t[16 * 64 + pov_square(ep, us).index()] = 1.0;
    }

    // Plane 17: halfmove clock / 100. Plane 18: all ones.
    t[17 * 64..18 * 64].fill(pos.halfmove_clock as f32 / 100.0);
    t[18 * 64..].fill(1.0);
    t
}

/// Policy-head index (0..4168) for `mv` played by `side`.
pub fn policy_index(mv: Move, side: Color) -> usize {
    let from = pov_square(mv.from(), side);
    let to = pov_square(mv.to(), side);
    let piece = match mv.promotion() {
        None | Some(PieceKind::Queen) => return from.index() * 64 + to.index(),
        Some(PieceKind::Knight) => 0,
        Some(PieceKind::Bishop) => 1,
        _ => 2,
    };
    let direction = match to.file() as i8 - from.file() as i8 {
        0 => 0,
        -1 => 1,
        _ => 2,
    };
    4096 + from.file() as usize * 9 + direction * 3 + piece
}

#[cfg(test)]
mod tests {
    use super::*;
    use gambito_engine::{fen, MoveFlags};

    fn sq(name: &str) -> Square {
        name.parse().unwrap()
    }

    #[test]
    fn startpos_looks_identical_from_both_sides() {
        // The start position is symmetric, so after the POV mirror the
        // tensor must not depend on who is to move.
        let white = fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let black = fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_eq!(encode_planes(&white), encode_planes(&black));
    }

    #[test]
    fn startpos_plane_shapes() {
        let t = encode_planes(&Position::startpos());
        // Our pawns: 8 ones, all on rank 2 (index 1).
        let pawns = &t[0..64];
        assert_eq!(pawns.iter().sum::<f32>(), 8.0);
        assert!(pawns[8..16].iter().all(|&v| v == 1.0));
        // All four castling planes lit, ep empty, clock 0, ones plane full.
        for p in 12..16 {
            assert!(t[p * 64..(p + 1) * 64].iter().all(|&v| v == 1.0));
        }
        assert_eq!(t[16 * 64..17 * 64].iter().sum::<f32>(), 0.0);
        assert!(t[17 * 64..18 * 64].iter().all(|&v| v == 0.0));
        assert!(t[18 * 64..].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn en_passant_square_is_mirrored_for_black() {
        // White just played f2f4; Black to move, ep target f3. In Black's
        // frame f3 mirrors to f6 = rank 5, file 5.
        let pos =
            fen::parse("rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2").unwrap();
        let t = encode_planes(&pos);
        let plane = &t[16 * 64..17 * 64];
        assert_eq!(plane.iter().sum::<f32>(), 1.0);
        assert_eq!(plane[5 * 8 + 5], 1.0);
    }

    #[test]
    fn halfmove_clock_plane_is_fractional() {
        let pos = fen::parse("8/8/4k3/8/8/4K3/8/R7 w - - 37 60").unwrap();
        let t = encode_planes(&pos);
        assert!(t[17 * 64..18 * 64].iter().all(|&v| v == 0.37));
    }

    // ---- policy_index spec (red until TODO(human) is filled in) ----

    #[test]
    fn plain_move_uses_from_64_plus_to() {
        // e2=12, e4=28 -> 12*64+28 = 796.
        let mv = Move::new(sq("e2"), sq("e4"), MoveFlags::DoublePush);
        assert_eq!(policy_index(mv, Color::White), 796);
    }

    #[test]
    fn black_mirror_move_shares_the_index() {
        // e7e5 by Black mirrors to e2e4: the same index, 796. This is the
        // whole point of the POV frame — one pattern learned once.
        let mv = Move::new(sq("e7"), sq("e5"), MoveFlags::DoublePush);
        assert_eq!(policy_index(mv, Color::Black), 796);
    }

    #[test]
    fn queen_promotion_stays_in_the_flat_range() {
        // a7=48, a8=56 -> 48*64+56 = 3128.
        let mv = Move::new(sq("a7"), sq("a8"), MoveFlags::promo(PieceKind::Queen, false));
        assert_eq!(policy_index(mv, Color::White), 3128);
    }

    #[test]
    fn underpromotions_use_the_tail_slots() {
        // a7a8=N push: 4096 + 0*9 + 0*3 + 0 = 4096.
        let push = Move::new(sq("a7"), sq("a8"), MoveFlags::promo(PieceKind::Knight, false));
        assert_eq!(policy_index(push, Color::White), 4096);
        // a7xb8=R, capture toward file+1: 4096 + 0*9 + 2*3 + 2 = 4104.
        let cap = Move::new(sq("a7"), sq("b8"), MoveFlags::promo(PieceKind::Rook, true));
        assert_eq!(policy_index(cap, Color::White), 4104);
    }

    #[test]
    fn black_underpromotion_is_mirrored_first() {
        // g2g1=B by Black mirrors to g7g8: 4096 + 6*9 + 0*3 + 1 = 4151.
        let mv = Move::new(sq("g2"), sq("g1"), MoveFlags::promo(PieceKind::Bishop, false));
        assert_eq!(policy_index(mv, Color::Black), 4151);
    }
}
