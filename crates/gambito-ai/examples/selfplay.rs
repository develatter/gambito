//! Self-play game generator: the data half of the AlphaZero loop.
//!
//!   cargo run --release -p gambito-ai --example selfplay -- \
//!       --out games.jsonl [--games N] [--sims N] [--threads N] [--seed N] [--model path.cyw]
//!
//! Each position visited is written as one JSON line:
//!   {"fen":"...","pi":"e2e4:40 g1f3:12 ...","z":1.0}
//! where pi is the root visit distribution of the search and z is the final
//! game result from the POV of the side to move at that position.
//!
//! Unlike arena play, the searches here mix Dirichlet noise into the root
//! priors and the early moves are SAMPLED from the visit distribution
//! (temperature), so every game is different — that variety is the
//! exploration the training data needs.

use gambito_ai::{Mcts, NnEval};
use gambito_engine::{fen, uci, Color, Game, GameStatus, Move};
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

const MAX_PLIES: usize = 300;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut games, mut sims, mut seed) = (200u32, 128u32, 1u64);
    let mut threads = std::thread::available_parallelism().map_or(4, |n| n.get().max(3) - 2);
    let mut out_path = String::from("selfplay.jsonl");
    let mut model_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut next = |what: &str| it.next().unwrap_or_else(|| panic!("{what} needs a value"));
        match a.as_str() {
            "--games" => games = next("--games").parse().expect("--games N"),
            "--sims" => sims = next("--sims").parse().expect("--sims N"),
            "--threads" => threads = next("--threads").parse().expect("--threads N"),
            "--seed" => seed = next("--seed").parse().expect("--seed N"),
            "--out" => out_path = next("--out").clone(),
            "--model" => model_path = Some(next("--model").clone()),
            other => panic!("unknown argument: {other}"),
        }
    }

    let eval = match &model_path {
        Some(p) => {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            NnEval::from_bytes(&bytes).unwrap_or_else(|e| panic!("{p}: {e}"))
        }
        None => NnEval::embedded(),
    };
    let file = std::fs::File::create(&out_path).unwrap_or_else(|e| panic!("{out_path}: {e}"));
    let writer = Mutex::new(std::io::BufWriter::new(file));
    let next_game = AtomicU32::new(0);
    let positions = AtomicU32::new(0);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut mcts = Mcts::default();
                loop {
                    let g = next_game.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        return;
                    }
                    let mut rng = fastrand::Rng::with_seed(seed.wrapping_mul(1_000_003) + g as u64);
                    let mut game = Game::new();
                    // (fen, pi-string, side to move) per recorded position.
                    let mut samples: Vec<(String, String, Color)> = Vec::new();
                    while !game.status().is_over() && game.moves_played().len() < MAX_PLIES {
                        let pos = game.position();
                        let Some(dist) =
                            mcts.search_visits(pos, &eval, sims, Some(&mut rng))
                        else {
                            break;
                        };
                        let pi = dist
                            .iter()
                            .filter(|(_, v)| *v > 0)
                            .map(|(mv, v)| format!("{}:{v}", uci::format(*mv)))
                            .collect::<Vec<_>>()
                            .join(" ");
                        samples.push((fen::format(pos), pi, pos.side_to_move));
                        let ply = game.moves_played().len();
                        game.play(sample_move(&dist, ply, &mut rng));
                    }
                    // Game result from White's POV; each sample flips to its mover's POV.
                    let white_z = match game.status() {
                        GameStatus::Checkmate { winner } => {
                            if winner == Color::White {
                                1.0f32
                            } else {
                                -1.0
                            }
                        }
                        _ => 0.0,
                    };
                    let mut lines = String::new();
                    for (fen, pi, side) in &samples {
                        let z = if *side == Color::White { white_z } else { -white_z };
                        lines.push_str(&format!("{{\"fen\":\"{fen}\",\"pi\":\"{pi}\",\"z\":{z}}}\n"));
                    }
                    let mut w = writer.lock().unwrap();
                    w.write_all(lines.as_bytes()).expect("write");
                    let total = positions.fetch_add(samples.len() as u32, Ordering::Relaxed)
                        + samples.len() as u32;
                    println!(
                        "game {:4}/{games}: {:24} in {:3} plies ({total} positions)",
                        g + 1,
                        status_label(game.status()),
                        samples.len(),
                    );
                }
            });
        }
    });
    writer.lock().unwrap().flush().expect("flush");
}

fn status_label(status: GameStatus) -> &'static str {
    match status {
        GameStatus::Checkmate { winner: Color::White } => "1-0 (checkmate)",
        GameStatus::Checkmate { winner: Color::Black } => "0-1 (checkmate)",
        GameStatus::Ongoing => "draw (max plies)",
        _ => "draw",
    }
}

/// Picks the move to actually play from the root visit distribution.
///
/// AlphaZero plays the first ~30 plies at temperature 1 (sample a move with
/// probability proportional to its visit count — variety) and the rest at
/// temperature ~0 (play the most-visited move — strength). Without the hot
/// phase every game from the same net would be identical; without the cold
/// phase endgames would be full of blunders and the z labels would be noise.
fn sample_move(dist: &[(Move, u32)], ply: usize, rng: &mut fastrand::Rng) -> Move {
    const HOT_PLIES: usize = 30;
    let total: u32 = dist.iter().map(|(_, v)| v).sum();
    if ply >= HOT_PLIES || total == 0 {
        // Cold phase: play the strongest move the search found.
        return dist.iter().max_by_key(|(_, v)| *v).unwrap().0;
    }
    // Hot phase: roulette-wheel over visit counts (temperature 1).
    let mut ticket = rng.u32(..total);
    for &(mv, v) in dist {
        if ticket < v {
            return mv;
        }
        ticket -= v;
    }
    dist[0].0
}
