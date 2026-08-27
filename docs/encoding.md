# Position & policy encoding contract

Single source of truth for how positions and moves are encoded as tensors.
`gambito-ai` (Rust, inference) and `python/gambito_train` (training) MUST both
implement exactly this. Golden-tensor tests on both sides guard the contract.

## Perspective

Everything is encoded **from the side to move's point of view**. If Black is
to move, mirror ranks (rank r becomes 7-r; files are NOT mirrored) and swap
piece colors. "Us" below always means the side to move. The value head and
policy indices live in this flipped frame too.

## Input planes — shape `[19, 8, 8]`, f32

Plane layout `[plane][rank][file]`, a1 = `[0][0]` in the us-POV frame.
Piece order matches `PieceKind::index()`: P, N, B, R, Q, K.

| Plane | Content                              |
|-------|--------------------------------------|
| 0–5   | Us: P, N, B, R, Q, K (one-hot)       |
| 6–11  | Them: P, N, B, R, Q, K (one-hot)     |
| 12    | Us can castle king-side (all 0/1)    |
| 13    | Us can castle queen-side (all 0/1)   |
| 14    | Them can castle king-side (all 0/1)  |
| 15    | Them can castle queen-side (all 0/1) |
| 16    | En-passant target square (one-hot)   |
| 17    | Halfmove clock / 100 (constant)      |
| 18    | All ones (edge detector aid)         |

## Policy index — 4,168 logits

Moves are indexed in the us-POV frame (squares already flipped):

- **0..4095**: `from * 64 + to`. Queen promotions use this range too.
- **4096..4167**: underpromotions: `4096 + from_file * 9 + direction * 3 + piece`
  where `direction` is 0 = push, 1 = capture toward file-1, 2 = capture toward
  file+1, and `piece` is 0 = N, 1 = B, 2 = R.

Illegal-move logits are masked to -inf before softmax; priors are the softmax
over legal moves only.

## Value

Scalar in `[-1, 1]`, from the side to move's perspective: +1 = side to move
wins, 0 = draw. Training targets use game outcome z; MCTS backs up values
negated at each ply.

## Model file (`model.cyw`, later phase)

Magic `CYW1`, then per-layer: i8 weight blob + f32 scale + i32 bias. Defined
in detail when the export lands; this section is a placeholder on purpose.
