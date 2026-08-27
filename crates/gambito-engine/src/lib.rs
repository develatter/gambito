//! Chess rules, move generation and notation. Zero external dependencies.
//!
//! The crate is IO-free and deterministic so it can be reused verbatim by the
//! TUI, the AI search, the network layer and (later) Python bindings.

mod attacks;
mod bitboard;
mod fen;
mod game;
mod movegen;
mod moves;
mod perft;
mod position;
mod san;
mod types;
mod uci;
mod zobrist;

pub use bitboard::Bitboard;
pub use fen::FenError;
pub use game::{Game, GameStatus};
pub use moves::{Move, MoveFlags, MoveList};
pub use perft::perft;
pub use position::Position;
pub use types::{CastlingRights, Color, Piece, PieceKind, Square};
