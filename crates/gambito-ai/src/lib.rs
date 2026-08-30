//! Search and evaluation for gambito.
//!
//! The `Brain` trait is the seam every opponent implements — MCTS+NN now,
//! NNUE+alpha-beta someday. `Evaluator` is the inner seam MCTS searches
//! through: MaterialEval today, the int8 network when training lands (M2).

mod brain;
pub mod encode;
mod eval;
mod mcts;
pub mod nn;

pub use brain::{Brain, MctsBrain, RandomBrain};
pub use encode::{encode_planes, policy_index, PLANE_COUNT, POLICY_SIZE};
pub use eval::{Evaluator, MaterialEval};
pub use mcts::Mcts;
pub use nn::{Network, NnEval};
