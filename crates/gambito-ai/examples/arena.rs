//! Head-to-head match between two .cyw models (or a model and the embedded
//! one): the AlphaZero gatekeeper. A candidate replaces the champion only
//! if it wins the arena.
//!
//!   cargo run --release -p gambito-ai --example arena -- \
//!       candidate.cyw [champion.cyw] [--games N] [--sims N]
//!
//! With one path, the opponent is the model embedded in the binary.
//!
//! MCTS is deterministic, so equal games would repeat move for move.
//! Games are played in PAIRS instead: each pair starts from the same short
//! random opening (seeded, reproducible) and swaps colors, so an unbalanced
//! opening penalizes both models equally. Score and Elo estimate are from
//! the FIRST (candidate) model's point of view.

use gambito_ai::{Brain, MctsBrain, NnEval};
use gambito_engine::{Color, Game, GameStatus};

const MAX_PLIES: usize = 300;
const OPENING_PLIES: usize = 4;

fn brain(path: Option<&str>, sims: u32) -> MctsBrain<NnEval> {
    let eval = match path {
        Some(p) => {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            NnEval::from_bytes(&bytes).unwrap_or_else(|e| panic!("{p}: {e}"))
        }
        None => NnEval::embedded(),
    };
    MctsBrain::new(eval, sims)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut paths: Vec<&str> = Vec::new();
    let (mut games, mut sims) = (20u32, 100u32);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--games" => games = it.next().expect("--games N").parse().expect("--games N"),
            "--sims" => sims = it.next().expect("--sims N").parse().expect("--sims N"),
            p => paths.push(p),
        }
    }
    assert!(!paths.is_empty(), "usage: arena candidate.cyw [champion.cyw]");
    let candidate_path = paths[0];
    let champion_path = paths.get(1).copied();

    let (mut wins, mut draws, mut losses) = (0u32, 0u32, 0u32);
    for g in 0..games {
        let mut candidate = brain(Some(candidate_path), sims);
        let mut champion = brain(champion_path, sims);
        // Games come in pairs: same seeded random opening, colors swapped.
        let candidate_is_white = g % 2 == 0;
        let mut rng = fastrand::Rng::with_seed(1000 + (g / 2) as u64);
        let mut game = Game::new();
        for _ in 0..OPENING_PLIES {
            let moves = game.legal_moves();
            game.play(moves.as_slice()[rng.usize(..moves.len())]);
        }
        while !game.status().is_over() && game.moves_played().len() < MAX_PLIES {
            let white_to_move = game.position().side_to_move == Color::White;
            let mover =
                if white_to_move == candidate_is_white { &mut candidate } else { &mut champion };
            match mover.choose(&game) {
                Some(mv) => game.play(mv),
                None => break,
            }
        }
        let result = match game.status() {
            GameStatus::Checkmate { winner } => {
                if (winner == Color::White) == candidate_is_white {
                    wins += 1;
                    "1-0 (candidate)"
                } else {
                    losses += 1;
                    "0-1 (champion)"
                }
            }
            _ => {
                draws += 1;
                "draw"
            }
        };
        println!(
            "game {:2}/{games}: {result:18} in {:3} plies (candidate as {})",
            g + 1,
            game.moves_played().len(),
            if candidate_is_white { "White" } else { "Black" },
        );
    }

    let score = (wins as f64 + draws as f64 / 2.0) / games as f64;
    print!("\ncandidate {wins}W {draws}D {losses}L  score {:.1}%", score * 100.0);
    if score > 0.0 && score < 1.0 {
        println!("  (Elo {:+.0})", 400.0 * (score / (1.0 - score)).log10());
    } else {
        println!();
    }
}
