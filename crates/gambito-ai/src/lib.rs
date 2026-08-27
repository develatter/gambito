//! Search and evaluation for gambito.
//!
//! The `Brain` trait is the seam every opponent implements — MCTS+NN now,
//! NNUE+alpha-beta someday. `Evaluator` is the inner seam MCTS searches
//! through: MaterialEval today, the int8 network when training lands (M2).

mod brain;
mod eval;
mod mcts;

pub use brain::{Brain, MctsBrain, RandomBrain};
pub use eval::{Evaluator, MaterialEval};
pub use mcts::Mcts;
