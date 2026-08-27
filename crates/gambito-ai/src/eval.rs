use gambito_engine::{Move, PieceKind, Position};

/// Position evaluation as MCTS consumes it: a prior probability per legal
/// move (aligned with `moves`, should sum to ~1) and a value in [-1, 1]
/// from the side to move's perspective (+1 = side to move is winning).
///
/// This is the seam the int8 network will implement; see docs/encoding.md.
pub trait Evaluator {
    fn evaluate(&self, pos: &Position, moves: &[Move]) -> (Vec<f32>, f32);
}

/// Stand-in evaluator until the network lands: uniform priors plus a
/// material count. Enough for MCTS to punish hung pieces and find mates.
pub struct MaterialEval;

impl Evaluator for MaterialEval {
    fn evaluate(&self, pos: &Position, moves: &[Move]) -> (Vec<f32>, f32) {
        let priors = vec![1.0 / moves.len().max(1) as f32; moves.len()];
        (priors, material_value(pos))
    }
}

/// Value in [-1, 1] from the side to move's perspective, from material only.
/// tanh(diff/10) makes a queen-up position ~0.72 — clearly winning without
/// saturating, so the search still prefers converting to mate.
fn material_value(pos: &Position) -> f32 {
    const WEIGHTS: [(PieceKind, f32); 5] = [
        (PieceKind::Pawn, 1.0),
        (PieceKind::Knight, 3.0),
        (PieceKind::Bishop, 3.25),
        (PieceKind::Rook, 5.0),
        (PieceKind::Queen, 9.0),
    ];
    let mut diff = 0.0;
    for (kind, weight) in WEIGHTS {
        let us = pos.pieces(pos.side_to_move, kind).count() as f32;
        let them = pos.pieces(pos.side_to_move.opposite(), kind).count() as f32;
        diff += weight * (us - them);
    }
    (diff / 10.0).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gambito_engine::{fen, Position};

    #[test]
    fn startpos_is_balanced() {
        assert_eq!(material_value(&Position::startpos()), 0.0);
    }

    #[test]
    fn value_is_from_side_to_move_perspective() {
        // White is up a queen: White to move sees a positive value...
        let up = fen::parse("4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1").unwrap();
        // ...and from Black's seat the same material reads negative.
        let down = fen::parse("4k3/8/8/8/8/8/8/QQ2K3 b - - 0 1").unwrap();
        assert!(material_value(&up) > 0.05);
        assert!(material_value(&down) < -0.05);
        assert!((material_value(&up) + material_value(&down)).abs() < 1e-6);
    }

    #[test]
    fn value_stays_in_range() {
        // Nine queens up: still must squash inside [-1, 1].
        let crushed = fen::parse("4k3/8/8/8/QQ2QQ2/QQ1QQ3/8/Q3K3 w - - 0 1").unwrap();
        let v = material_value(&crushed);
        assert!(v > 0.5 && v <= 1.0, "got {v}");
    }
}
