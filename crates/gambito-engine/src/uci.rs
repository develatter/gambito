//! UCI long algebraic ("e2e4", "e7e8q") — the wire and storage move format.

use crate::movegen::legal_moves;
use crate::moves::Move;
use crate::position::Position;
use crate::types::{PieceKind, Square};

pub fn format(mv: Move) -> String {
    match mv.promotion() {
        Some(kind) => format!("{}{}{}", mv.from(), mv.to(), kind.to_char()),
        None => format!("{}{}", mv.from(), mv.to()),
    }
}

/// Resolves a UCI string against the legal moves of `pos`, so flags (castle,
/// en passant, capture) come out right without trusting the sender.
pub fn parse(pos: &Position, s: &str) -> Option<Move> {
    if s.len() < 4 || s.len() > 5 {
        return None;
    }
    let from: Square = s[0..2].parse().ok()?;
    let to: Square = s[2..4].parse().ok()?;
    let promo = match s.as_bytes().get(4) {
        Some(&c) => Some(PieceKind::from_char(c as char)?),
        None => None,
    };
    legal_moves(pos)
        .iter()
        .copied()
        .find(|mv| mv.from() == from && mv.to() == to && mv.promotion() == promo)
}
