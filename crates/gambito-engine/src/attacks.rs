//! Attack sets: const-generated leaper tables and ray-scan sliders.
//!
//! Sliders deliberately use a simple ray scan instead of magic bitboards; the
//! call sites go through these two functions only, so swapping in magics later
//! is a local change.

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

const fn shift(bb: u64, dfile: i8, drank: i8) -> u64 {
    // Masks off files that would wrap around the A/H edge.
    let masked = match dfile {
        -2 => bb & !0x0303_0303_0303_0303, // not files a,b
        -1 => bb & !0x0101_0101_0101_0101, // not file a
        0 => bb,
        1 => bb & !0x8080_8080_8080_8080, // not file h
        2 => bb & !0xC0C0_C0C0_C0C0_C0C0, // not files g,h
        _ => 0,
    };
    let offset = drank * 8 + dfile;
    if offset >= 0 {
        masked << offset
    } else {
        masked >> -offset
    }
}

const fn leaper_table(deltas: &[(i8, i8)]) -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0;
    while sq < 64 {
        let from = 1u64 << sq;
        let mut acc = 0u64;
        let mut i = 0;
        while i < deltas.len() {
            acc |= shift(from, deltas[i].0, deltas[i].1);
            i += 1;
        }
        table[sq] = Bitboard(acc);
        sq += 1;
    }
    table
}

pub const KNIGHT: [Bitboard; 64] = leaper_table(&[
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]);

pub const KING: [Bitboard; 64] = leaper_table(&[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]);

/// Squares a pawn of `color` on the indexed square attacks (captures only).
pub const PAWN_ATTACKS: [[Bitboard; 64]; 2] = [
    leaper_table(&[(-1, 1), (1, 1)]),  // White
    leaper_table(&[(-1, -1), (1, -1)]), // Black
];

#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT[sq.index()]
}

#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING[sq.index()]
}

#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.index()]
}

fn ray_attacks(sq: Square, occupied: Bitboard, directions: &[(i8, i8)]) -> Bitboard {
    let mut acc = Bitboard::EMPTY;
    for &(df, dr) in directions {
        let mut current = sq;
        while let Some(next) = current.offset(df, dr) {
            acc.set(next);
            if occupied.contains(next) {
                break;
            }
            current = next;
        }
    }
    acc
}

pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &[(-1, -1), (-1, 1), (1, -1), (1, 1)])
}

pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &[(-1, 0), (1, 0), (0, -1), (0, 1)])
}

pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_corner_and_center() {
        assert_eq!(knight_attacks(Square::A1).count(), 2);
        assert_eq!(knight_attacks("d4".parse().unwrap()).count(), 8);
    }

    #[test]
    fn king_edges() {
        assert_eq!(king_attacks(Square::A1).count(), 3);
        assert_eq!(king_attacks("e4".parse().unwrap()).count(), 8);
    }

    #[test]
    fn pawn_attack_direction() {
        let e4: Square = "e4".parse().unwrap();
        let white = pawn_attacks(Color::White, e4);
        assert!(white.contains("d5".parse().unwrap()));
        assert!(white.contains("f5".parse().unwrap()));
        let black = pawn_attacks(Color::Black, e4);
        assert!(black.contains("d3".parse().unwrap()));
    }

    #[test]
    fn rook_blocked_by_occupancy() {
        let e4: Square = "e4".parse().unwrap();
        let blocker: Square = "e6".parse().unwrap();
        let attacks = rook_attacks(e4, Bitboard::from_square(blocker));
        assert!(attacks.contains(blocker));
        assert!(!attacks.contains("e7".parse().unwrap()));
        assert!(attacks.contains("a4".parse().unwrap()));
    }
}
