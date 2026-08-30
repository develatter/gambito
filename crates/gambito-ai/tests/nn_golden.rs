//! Cross-language golden test for the network inference: the values and
//! priors PyTorch computed with the dequantized weights (nn_golden.txt,
//! written by `python -m gambito_train.export`) must match the hand-rolled
//! Rust forward pass. If this fails, the Rust math drifted from
//! export.py::forward_dequant — fix the drift, never the golden file.

use gambito_ai::{encode_planes, Evaluator, NnEval};
use gambito_engine::{fen, legal_moves, uci};

const GOLDEN: &str = include_str!("nn_golden.txt");
const TOL: f32 = 2e-3;

fn parse_line(line: &str) -> (String, f32, Vec<(String, f32)>) {
    let mut parts = line.splitn(3, " | ");
    let fen = parts.next().unwrap().to_string();
    let value: f32 = parts.next().unwrap().parse().unwrap();
    let priors = parts
        .next()
        .unwrap()
        .split_whitespace()
        .map(|pair| {
            let (mv, p) = pair.split_once(':').unwrap();
            (mv.to_string(), p.parse().unwrap())
        })
        .collect();
    (fen, value, priors)
}

#[test]
fn values_match_pytorch() {
    let eval = NnEval::embedded();
    for line in GOLDEN.lines() {
        let (fen_str, want, _) = parse_line(line);
        let pos = fen::parse(&fen_str).unwrap();
        let (_, got) = eval.network().forward(&encode_planes(&pos));
        assert!((got - want).abs() < TOL, "{fen_str}: value {got} != {want}");
    }
}

#[test]
fn priors_match_pytorch() {
    let eval = NnEval::embedded();
    for line in GOLDEN.lines() {
        let (fen_str, _, want) = parse_line(line);
        let pos = fen::parse(&fen_str).unwrap();
        let moves: Vec<_> = legal_moves(&pos).iter().copied().collect();
        let (priors, _) = eval.evaluate(&pos, &moves);
        assert_eq!(priors.len(), moves.len());
        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{fen_str}: priors sum to {sum}");
        for (mv, want_p) in &want {
            let i = moves.iter().position(|m| &uci::format(*m) == mv).unwrap();
            assert!(
                (priors[i] - want_p).abs() < TOL,
                "{fen_str} {mv}: prior {} != {want_p}",
                priors[i]
            );
        }
    }
}
