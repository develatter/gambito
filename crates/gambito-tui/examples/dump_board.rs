//! Dev aid: renders the board to a TestBackend and prints the raw symbols,
//! so sprite silhouettes can be eyeballed without a live terminal.

use gambito_engine::{Bitboard, Position};
use gambito_tui::debug_board_dump;

fn main() {
    let pos = Position::startpos();
    print!("{}", debug_board_dump(&pos, 100, 45));
    let _ = Bitboard::EMPTY;
}
