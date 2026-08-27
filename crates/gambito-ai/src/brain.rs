use crate::eval::Evaluator;
use crate::mcts::Mcts;
use gambito_engine::{Game, Move};

/// An opponent. Takes the whole game (not just the position) so future
/// brains can reason about repetition and clocks.
pub trait Brain: Send {
    fn choose(&mut self, game: &Game) -> Option<Move>;
    /// Shown in the UI.
    fn name(&self) -> &str;
}

/// Uniform random legal mover; the baseline every other brain must beat.
pub struct RandomBrain {
    rng: fastrand::Rng,
}

impl RandomBrain {
    pub fn new(seed: u64) -> RandomBrain {
        RandomBrain { rng: fastrand::Rng::with_seed(seed) }
    }
}

impl Brain for RandomBrain {
    fn choose(&mut self, game: &Game) -> Option<Move> {
        let moves = game.legal_moves();
        if moves.is_empty() {
            return None;
        }
        Some(moves.as_slice()[self.rng.usize(..moves.len())])
    }

    fn name(&self) -> &str {
        "Random"
    }
}

/// PUCT search over a fixed simulation budget; difficulty is the budget.
pub struct MctsBrain<E: Evaluator> {
    mcts: Mcts,
    evaluator: E,
    simulations: u32,
}

impl<E: Evaluator> MctsBrain<E> {
    pub fn new(evaluator: E, simulations: u32) -> MctsBrain<E> {
        MctsBrain { mcts: Mcts::default(), evaluator, simulations }
    }
}

impl<E: Evaluator + Send> Brain for MctsBrain<E> {
    fn choose(&mut self, game: &Game) -> Option<Move> {
        self.mcts.search(game.position(), &self.evaluator, self.simulations)
    }

    fn name(&self) -> &str {
        "MCTS"
    }
}
