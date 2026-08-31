# Progress

Terminal chess in Rust with an AlphaZero-lite AI, built as a learning
project. Four milestones: engine + TUI, neural-network AI, online blitz,
correspondence play. This file tracks what is done and what remains.

## Done

### M1 — Engine and TUI

- Bitboard move generator, perft-verified; FEN and UCI parsing/formatting.
- `Game` layer with full end-of-game detection (checkmate, stalemate,
  fifty-move rule, repetition, insufficient material).
- Ratatui hotseat TUI: 8-bit sprite board, menus, move input. The AI
  thinks on a worker thread so the human's move always renders first.

### M2 — AlphaZero-lite AI

**Seams.** `Brain` (opponent) and `Evaluator` (position scorer) traits
separate search from evaluation. PUCT MCTS with no rollouts; simulation
budget is the difficulty dial. "Play the AI" menu entry (400 sims).

**Encoding contract** (`docs/encoding.md`). 19 input planes [19,8,8]
from the side to move's POV, 4168-way policy indexing, value in [-1,1].
Implemented twice — Rust (`gambito-ai/src/encode.rs`) and Python
(`gambito_train/encoding.py`) — and locked by cross-language golden
tests generated from Rust and verified in pytest.

**Network.** GambitoNet: 64-channel stem, 5 residual blocks, policy and
value heads; 1,485,857 parameters (~1.5 MB as int8).

**Supervised bootstrap.** Lichess Elite PGNs (2500+ vs 2300+, filtered
to 1800+ base 3min+) streamed into compact .npz datasets; training on an
RTX 5070 at 74–87k pos/s. Full-month run: 280k games → 24.6M samples,
6 epochs, val top-1 move accuracy 40.3% → **47.1%**.

**Deployment.** Custom `.cyw` int8 format (BN folded, per-layer
symmetric quantization, spec in `docs/encoding.md`); hand-rolled Rust
inference (padded, unrolled convolutions, ~5 ms/forward on one core) —
no runtime dependencies. The model is embedded in the binary via
`include_bytes!` (2.1 MB total). PyTorch-vs-Rust golden tests pin values
and priors to 2e-3.

**Arena gatekeeping.** Candidates must beat the embedded champion in
paired games from seeded random openings with colors swapped (MCTS is
deterministic, so unpaired games repeat move for move). The full-month
net beat the 4.4M bootstrap net 8W 9D 3L (Elo +89) and was crowned.

**Self-play loop (generation 1).** The full AlphaZero loop exists:

- Rust generator (`examples/selfplay.rs`): threaded games, Dirichlet(0.3)
  noise at the root (ε 0.25), temperature-1 sampling for 30 plies then
  argmax, JSONL output of (fen, root visit distribution π, outcome z).
- Soft-target training: unified cross-entropy over visit distributions
  (the supervised case is the one-hot special case, verified equal to the
  old loss), `--init` fine-tuning, replay-buffer mixing of multiple .npz.
- Verdict: 1,500 games (128 sims, 141k positions) fine-tuned pure lost
  the arena 0W 15D 5L (Elo −89, catastrophic forgetting); mixed with a
  400k-sample supervised anchor it recovered to 3W 13D 4L (Elo −17).
  **The champion stands.** At this scale self-play needs the supervised
  anchor and considerably more games to win — the gatekeeper did its job.

**Strength estimate.** No absolute measurement yet. By accuracy
triangulation against the Maia models, roughly 1400–1600 Lichess Elo
with wide error bars.

## Pending

### M2 follow-ups (optional — the AI works)

- **Self-play generation 2**: ~6,000 games overnight with the existing
  pipeline; repeat pack → mixed fine-tune → arena.
- **MCTS tree reuse**: keep the subtree between moves instead of
  discarding it — free strength at the same per-move cost.
- **Absolute Elo measurement**: a UCI bridge to Stockfish with
  `UCI_LimitStrength`/`UCI_Elo` at several levels to find the 50% point.

### M3 — Online blitz

- `gambito-proto` / `gambito-net` crates, websocket transport, clocks,
  networked games from the TUI.

### M4 — Correspondence

- P2P play with SQLite-persisted long-running games.

## Conventions

- Training data, checkpoints, and arena artifacts live outside the repo
  in `~/gambito-data/`; only the crowned `.cyw` model is committed.
- Work happens in git worktrees, fast-forward merged to `main`; `main`
  always compiles.
