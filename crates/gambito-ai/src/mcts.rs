//! PUCT Monte-Carlo tree search, AlphaZero style: no rollouts, the
//! evaluator scores leaves directly. Copy-make positions live in an arena.

use crate::eval::Evaluator;
use gambito_engine::{legal_moves, Move, Position};

struct Node {
    pos: Position,
    moves: Vec<Move>,
    priors: Vec<f32>,
    /// Per-edge child node index (usize::MAX = not expanded yet).
    children: Vec<usize>,
    /// Per-edge visit count and accumulated value (from this node's POV).
    visits: Vec<u32>,
    totals: Vec<f32>,
    /// Terminal value from this node's side-to-move POV, if game over here.
    terminal: Option<f32>,
}

const UNEXPANDED: usize = usize::MAX;

pub struct Mcts {
    pub c_puct: f32,
    nodes: Vec<Node>,
}

impl Default for Mcts {
    fn default() -> Mcts {
        Mcts { c_puct: 1.5, nodes: Vec::new() }
    }
}

impl Mcts {
    /// Runs `simulations` PUCT iterations and returns the most-visited root
    /// move (None only when the position has no legal moves).
    pub fn search<E: Evaluator>(
        &mut self,
        root: &Position,
        evaluator: &E,
        simulations: u32,
    ) -> Option<Move> {
        self.nodes.clear();
        self.new_node(root.clone(), evaluator);
        if self.nodes[0].moves.is_empty() {
            return None;
        }
        for _ in 0..simulations {
            self.simulate(evaluator);
        }
        let root = &self.nodes[0];
        let best = (0..root.moves.len()).max_by_key(|&i| root.visits[i])?;
        Some(root.moves[best])
    }

    fn new_node<E: Evaluator>(&mut self, pos: Position, evaluator: &E) -> (usize, f32) {
        let moves = legal_moves(&pos);
        let moves: Vec<Move> = moves.iter().copied().collect();
        let (priors, value, terminal) = if moves.is_empty() {
            // Mate is bad for the side to move; stalemate is a draw.
            let v = if pos.in_check(pos.side_to_move) { -1.0 } else { 0.0 };
            (Vec::new(), v, Some(v))
        } else if pos.halfmove_clock >= 100 {
            (Vec::new(), 0.0, Some(0.0))
        } else {
            let (priors, value) = evaluator.evaluate(&pos, &moves);
            (priors, value, None)
        };
        let n = moves.len();
        self.nodes.push(Node {
            pos,
            moves: if terminal.is_some() { Vec::new() } else { moves },
            priors,
            children: vec![UNEXPANDED; n],
            visits: vec![0; n],
            totals: vec![0.0; n],
            terminal,
        });
        (self.nodes.len() - 1, value)
    }

    fn simulate<E: Evaluator>(&mut self, evaluator: &E) {
        // Walk down by PUCT until reaching a terminal node or a fresh edge.
        let mut path: Vec<(usize, usize)> = Vec::new(); // (node, edge)
        let mut node = 0usize;
        let leaf_value = loop {
            if let Some(v) = self.nodes[node].terminal {
                break v;
            }
            let edge = self.select_edge(node);
            path.push((node, edge));
            let child = self.nodes[node].children[edge];
            if child == UNEXPANDED {
                let next = self.nodes[node].pos.apply(self.nodes[node].moves[edge]);
                let (idx, value) = self.new_node(next, evaluator);
                self.nodes[node].children[edge] = idx;
                break value;
            }
            node = child;
        };

        // Back up, flipping sign each ply: a good child value is bad for the
        // parent's mover. leaf_value is from the leaf's side-to-move POV.
        let mut value = -leaf_value;
        for &(node, edge) in path.iter().rev() {
            let n = &mut self.nodes[node];
            n.visits[edge] += 1;
            n.totals[edge] += value;
            value = -value;
        }
    }

    fn select_edge(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        let total_visits: u32 = n.visits.iter().sum();
        let sqrt_total = ((total_visits + 1) as f32).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n.moves.len() {
            let q = if n.visits[i] == 0 { 0.0 } else { n.totals[i] / n.visits[i] as f32 };
            let u = self.c_puct * n.priors[i] * sqrt_total / (1 + n.visits[i]) as f32;
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::MaterialEval;
    use gambito_engine::fen;

    #[test]
    fn finds_mate_in_one() {
        // Ra8# is the only mating move.
        let pos = fen::parse("4k3/8/4K3/8/8/8/8/R7 w - - 0 1").unwrap();
        let mv = Mcts::default().search(&pos, &MaterialEval, 400).unwrap();
        assert_eq!(gambito_engine::uci::format(mv), "a1a8");
    }

    #[test]
    fn avoids_stalemate_when_winning() {
        // Kf6 or Qg7# win; Qf7?? would be stalemate. With terminal values
        // wired correctly the search never prefers the stalemate.
        let pos = fen::parse("7k/8/5K2/6Q1/8/8/8/8 w - - 0 1").unwrap();
        let mv = Mcts::default().search(&pos, &MaterialEval, 800).unwrap();
        assert_ne!(gambito_engine::uci::format(mv), "g5f7");
    }

    #[test]
    fn returns_none_when_no_moves() {
        let pos = fen::parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap(); // stalemate
        assert!(Mcts::default().search(&pos, &MaterialEval, 10).is_none());
    }
}
