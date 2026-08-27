use crate::fen::{self, FenError};
use crate::movegen::legal_moves;
use crate::moves::{Move, MoveList};
use crate::position::Position;
use crate::types::{Color, PieceKind};
use crate::{san, uci};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameStatus {
    Ongoing,
    Checkmate { winner: Color },
    Stalemate,
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
}

impl GameStatus {
    pub fn is_over(self) -> bool {
        self != GameStatus::Ongoing
    }
}

/// One played move with everything the UI and storage need to show it.
#[derive(Clone, Debug)]
pub struct PlayedMove {
    pub mv: Move,
    pub san: String,
    pub uci: String,
}

/// A full game: current position plus history for undo and repetition.
#[derive(Clone)]
pub struct Game {
    positions: Vec<Position>,
    played: Vec<PlayedMove>,
}

impl Game {
    pub fn new() -> Game {
        Game {
            positions: vec![Position::startpos()],
            played: Vec::new(),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Game, FenError> {
        Ok(Game {
            positions: vec![fen::parse(fen)?],
            played: Vec::new(),
        })
    }

    pub fn position(&self) -> &Position {
        self.positions
            .last()
            .expect("at least the initial position")
    }

    pub fn moves_played(&self) -> &[PlayedMove] {
        &self.played
    }

    pub fn legal_moves(&self) -> MoveList {
        legal_moves(self.position())
    }

    pub fn play(&mut self, mv: Move) {
        let pos = self.position();
        let played = PlayedMove {
            mv,
            san: san::format(pos, mv),
            uci: uci::format(mv),
        };
        self.positions.push(pos.apply(mv));
        self.played.push(played);
    }

    pub fn play_san(&mut self, input: &str) -> Option<Move> {
        let mv = san::parse(self.position(), input)?;
        self.play(mv);
        Some(mv)
    }

    pub fn play_uci(&mut self, input: &str) -> Option<Move> {
        let mv = uci::parse(self.position(), input)?;
        self.play(mv);
        Some(mv)
    }

    /// Undoes the last move; returns false at the initial position.
    pub fn undo(&mut self) -> bool {
        if self.played.is_empty() {
            return false;
        }
        self.positions.pop();
        self.played.pop();
        true
    }

    pub fn status(&self) -> GameStatus {
        let pos = self.position();
        if self.legal_moves().is_empty() {
            return if pos.in_check(pos.side_to_move) {
                GameStatus::Checkmate {
                    winner: pos.side_to_move.opposite(),
                }
            } else {
                GameStatus::Stalemate
            };
        }
        if pos.halfmove_clock >= 100 {
            return GameStatus::FiftyMoveRule;
        }
        if self.is_threefold() {
            return GameStatus::ThreefoldRepetition;
        }
        if is_insufficient_material(pos) {
            return GameStatus::InsufficientMaterial;
        }
        GameStatus::Ongoing
    }

    /// Current position occurred three times. Comparing hashes is enough for
    /// our purposes (collisions are ~2^-64); only positions since the last
    /// irreversible move can repeat, which the halfmove clock bounds.
    fn is_threefold(&self) -> bool {
        let current = self.position();
        let window = (current.halfmove_clock as usize).min(self.positions.len() - 1);
        let mut count = 1;
        for pos in self.positions.iter().rev().skip(1).take(window) {
            if pos.hash == current.hash {
                count += 1;
                if count >= 3 {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for Game {
    fn default() -> Game {
        Game::new()
    }
}

/// Neither side can possibly deliver mate (dead position by material).
fn is_insufficient_material(pos: &Position) -> bool {
    // Any pawn, rook or queen on the board means mate is still possible.
    for color in [Color::White, Color::Black] {
        for kind in [PieceKind::Pawn, PieceKind::Rook, PieceKind::Queen] {
            if !pos.pieces(color, kind).is_empty() {
                return false;
            }
        }
    }

    let bishops = pos.pieces(Color::White, PieceKind::Bishop).count()
        + pos.pieces(Color::Black, PieceKind::Bishop).count();
    let knights = pos.pieces(Color::White, PieceKind::Knight).count()
        + pos.pieces(Color::Black, PieceKind::Knight).count();

    if bishops + knights <= 1 {
        return true; // K vs K, K+B vs K, K+N vs K
    }
    if knights > 0 {
        return false; // knight plus anything else can still mate
    }
    // Only bishops left: dead if they all live on one square color.
    let mut light = false;
    let mut dark = false;
    for color in [Color::White, Color::Black] {
        for sq in pos.pieces(color, PieceKind::Bishop) {
            match (sq.file() + sq.rank()) % 2 {
                1 => light = true,
                _ => dark = true,
            }
        }
    }
    !(light && dark)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fools_mate() {
        let mut game = Game::new();
        for m in ["f3", "e5", "g4", "Qh4#"] {
            game.play_san(m)
                .unwrap_or_else(|| panic!("{m} should be legal"));
        }
        assert_eq!(
            game.status(),
            GameStatus::Checkmate {
                winner: Color::Black
            }
        );
    }

    #[test]
    fn stalemate_detected() {
        let game = Game::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(game.status(), GameStatus::Stalemate);
    }

    #[test]
    fn undo_restores_position() {
        let mut game = Game::new();
        let before = game.position().hash;
        game.play_san("e4").unwrap();
        assert!(game.undo());
        assert_eq!(game.position().hash, before);
        assert!(!game.undo());
    }

    #[test]
    fn threefold_by_shuffling() {
        let mut game = Game::new();
        for m in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"] {
            game.play_san(m).unwrap();
        }
        assert_eq!(game.status(), GameStatus::ThreefoldRepetition);
    }

    #[test]
    fn insufficient_material_cases() {
        // K vs K, K+N vs K, and same-colored bishops are dead.
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1", // f8 and c1 are both dark
        ] {
            let game = Game::from_fen(fen).unwrap();
            assert_eq!(game.status(), GameStatus::InsufficientMaterial, "{fen}");
        }
        // Opposite-colored bishops and two knights can still mate.
        for fen in [
            "4kb2/8/8/8/8/8/8/3BK3 w - - 0 1", // f8 dark, d1 light
            "4k3/8/8/8/8/8/8/3NKN2 w - - 0 1",
        ] {
            let game = Game::from_fen(fen).unwrap();
            assert_eq!(game.status(), GameStatus::Ongoing, "{fen}");
        }
    }

    #[test]
    fn fifty_move_rule() {
        let game = Game::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 80").unwrap();
        assert_eq!(game.status(), GameStatus::FiftyMoveRule);
    }
}
