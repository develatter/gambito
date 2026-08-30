//! Dumps the golden tensors that pin the Rust <-> Python encoding contract.
//! Regenerate after any encoding change with:
//!
//!   cargo run -p gambito-ai --example dump_golden > python/tests/golden/positions.json

use gambito_ai::{encode_planes, policy_index};
use gambito_engine::{fen, legal_moves, uci};
use std::fmt::Write;

const FENS: [&str; 7] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2",
    "2r5/1P6/8/8/7k/8/8/5K2 w - - 0 1",
    "5k2/8/8/8/8/8/6p1/5K2 b - - 0 1",
    "8/8/4k3/8/8/4K3/8/R7 w - - 37 60",
];

fn main() {
    let mut out = String::from("[\n");
    for (i, f) in FENS.iter().enumerate() {
        let pos = fen::parse(f).expect("golden FEN must parse");
        write!(out, "  {{\n    \"fen\": \"{f}\",\n    \"planes\": [").unwrap();
        for (j, v) in encode_planes(&pos).iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            write!(out, "{v}").unwrap();
        }
        out.push_str("],\n    \"moves\": {");
        for (j, mv) in legal_moves(&pos).iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            write!(out, "\"{}\": {}", uci::format(*mv), policy_index(*mv, pos.side_to_move))
                .unwrap();
        }
        out.push_str("}\n  }");
        out.push_str(if i + 1 < FENS.len() { ",\n" } else { "\n" });
    }
    out.push_str("]\n");
    print!("{out}");
}
