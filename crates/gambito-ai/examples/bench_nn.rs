//! Quick timing of the embedded network: raw forward pass and full MCTS
//! moves at a few simulation budgets. Run with --release.

use gambito_ai::{Brain, MctsBrain, NnEval};
use gambito_engine::{Game, Position};
use std::time::Instant;

fn main() {
    let eval = NnEval::embedded();
    let planes = gambito_ai::encode_planes(&Position::startpos());
    let net = eval.network();
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(net.forward(&planes));
    }
    println!("forward: {:?} per position", t.elapsed() / n);

    for sims in [100u32, 200, 400] {
        let mut brain = MctsBrain::new(NnEval::embedded(), sims);
        let game = Game::new();
        let t = Instant::now();
        let mv = brain.choose(&game).unwrap();
        println!("{sims} sims -> {:?} ({})", t.elapsed(), gambito_engine::uci::format(mv));
    }
}
