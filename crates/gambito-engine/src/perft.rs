//! Perft: exhaustive move-generation node counts, the engine's ground truth.

use crate::movegen::legal_moves;
use crate::position::Position;

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves.iter().map(|&mv| perft(&pos.apply(mv), depth - 1)).sum()
}

/// Per-root-move breakdown, invaluable when a perft count mismatches
/// (diff against `stockfish "go perft N"`).
pub fn perft_divide(pos: &Position, depth: u32) -> Vec<(String, u64)> {
    legal_moves(pos)
        .iter()
        .map(|&mv| (crate::uci::format(mv), perft(&pos.apply(mv), depth.saturating_sub(1))))
        .collect()
}
