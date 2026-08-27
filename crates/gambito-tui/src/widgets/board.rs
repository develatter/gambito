//! Board rendering and mouse hit-testing. Geometry lives in the SQUARE_W/H
//! consts so the mouse math and the renderer can never disagree.

use gambito_engine::{Bitboard, Color as Side, PieceKind, Position, Square};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

pub const SQUARE_W: u16 = 4;
pub const SQUARE_H: u16 = 2;
/// Rank labels to the left of the squares.
const MARGIN_X: u16 = 2;
pub const BOARD_W: u16 = MARGIN_X + 8 * SQUARE_W;
pub const BOARD_H: u16 = 8 * SQUARE_H + 1; // +1 for file labels

const LIGHT_SQ: Color = Color::Rgb(240, 217, 181);
const DARK_SQ: Color = Color::Rgb(181, 136, 99);
const SELECTED: Color = Color::Rgb(130, 151, 105);
const LAST_MOVE_LIGHT: Color = Color::Rgb(205, 210, 106);
const LAST_MOVE_DARK: Color = Color::Rgb(170, 162, 58);
const CHECK: Color = Color::Rgb(212, 83, 71);
const WHITE_PIECE: Color = Color::Rgb(255, 255, 255);
const BLACK_PIECE: Color = Color::Rgb(20, 18, 16);
const LABEL: Color = Color::Rgb(140, 140, 140);

pub struct BoardWidget<'a> {
    pub pos: &'a Position,
    pub flipped: bool,
    pub ascii: bool,
    pub selected: Option<Square>,
    pub targets: Bitboard,
    pub last_move: Option<(Square, Square)>,
    pub check: Option<Square>,
}

/// Maps a terminal cell inside `area` back to a board square.
pub fn square_at(area: Rect, column: u16, row: u16, flipped: bool) -> Option<Square> {
    let bx = area.x + MARGIN_X;
    if column < bx || row < area.y {
        return None;
    }
    let file_disp = (column - bx) / SQUARE_W;
    let rank_disp = (row - area.y) / SQUARE_H;
    if file_disp > 7 || rank_disp > 7 || row >= area.y + 8 * SQUARE_H {
        return None;
    }
    let file = if flipped { 7 - file_disp } else { file_disp };
    let rank = if flipped { rank_disp } else { 7 - rank_disp };
    Some(Square::new(file as u8, rank as u8))
}

fn glyph(kind: PieceKind, ascii: bool, side: Side) -> char {
    if ascii {
        let c = kind.to_char();
        return if side == Side::White { c.to_ascii_uppercase() } else { c };
    }
    // Filled glyphs for both sides; the fg color carries the side.
    match kind {
        PieceKind::King => '♚',
        PieceKind::Queen => '♛',
        PieceKind::Rook => '♜',
        PieceKind::Bishop => '♝',
        PieceKind::Knight => '♞',
        PieceKind::Pawn => '♟',
    }
}

impl Widget for BoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < BOARD_W || area.height < BOARD_H {
            return;
        }
        let bx = area.x + MARGIN_X;
        for rank_disp in 0..8u16 {
            for file_disp in 0..8u16 {
                let file = if self.flipped { 7 - file_disp } else { file_disp } as u8;
                let rank = if self.flipped { rank_disp } else { 7 - rank_disp } as u8;
                let sq = Square::new(file, rank);
                let dark = (file + rank) % 2 == 0;

                let mut bg = if dark { DARK_SQ } else { LIGHT_SQ };
                if let Some((from, to)) = self.last_move {
                    if sq == from || sq == to {
                        bg = if dark { LAST_MOVE_DARK } else { LAST_MOVE_LIGHT };
                    }
                }
                if self.selected == Some(sq) {
                    bg = SELECTED;
                }
                if self.check == Some(sq) {
                    bg = CHECK;
                }

                let x0 = bx + file_disp * SQUARE_W;
                let y0 = area.y + rank_disp * SQUARE_H;
                for dy in 0..SQUARE_H {
                    for dx in 0..SQUARE_W {
                        let cell = &mut buf[(x0 + dx, y0 + dy)];
                        cell.set_char(' ');
                        cell.set_bg(bg);
                    }
                }

                if let Some(piece) = self.pos.piece_at(sq) {
                    let fg = if piece.color == Side::White { WHITE_PIECE } else { BLACK_PIECE };
                    let cell = &mut buf[(x0 + 1, y0)];
                    cell.set_char(glyph(piece.kind, self.ascii, piece.color));
                    cell.set_fg(fg);
                } else if self.targets.contains(sq) {
                    let cell = &mut buf[(x0 + 1, y0)];
                    cell.set_char('•');
                    cell.set_fg(if dark { LIGHT_SQ } else { DARK_SQ });
                }
                // Capture targets keep the piece but get a marker in the corner.
                if self.targets.contains(sq) && self.pos.piece_at(sq).is_some() {
                    let cell = &mut buf[(x0 + SQUARE_W - 1, y0)];
                    cell.set_char('x');
                    cell.set_fg(CHECK);
                }
            }
            let rank_label = if self.flipped { rank_disp + 1 } else { 8 - rank_disp };
            let cell = &mut buf[(area.x, area.y + rank_disp * SQUARE_H)];
            cell.set_char((b'0' + rank_label as u8) as char);
            cell.set_fg(LABEL);
        }
        for file_disp in 0..8u16 {
            let file = if self.flipped { 7 - file_disp } else { file_disp };
            let cell = &mut buf[(bx + file_disp * SQUARE_W + 1, area.y + 8 * SQUARE_H)];
            cell.set_char((b'a' + file as u8) as char);
            cell.set_fg(LABEL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::new(0, 0, BOARD_W, BOARD_H)
    }

    #[test]
    fn hit_test_corners() {
        // Top-left square is a8 in normal orientation, h1 flipped.
        assert_eq!(square_at(area(), MARGIN_X, 0, false), Some("a8".parse().unwrap()));
        assert_eq!(square_at(area(), MARGIN_X, 0, true), Some("h1".parse().unwrap()));
        // Bottom-right of the grid.
        let col = MARGIN_X + 8 * SQUARE_W - 1;
        let row = 8 * SQUARE_H - 1;
        assert_eq!(square_at(area(), col, row, false), Some("h1".parse().unwrap()));
        // Label row and left margin miss.
        assert_eq!(square_at(area(), MARGIN_X, 8 * SQUARE_H, false), None);
        assert_eq!(square_at(area(), 0, 0, false), None);
    }

    #[test]
    fn hit_test_inverts_render_geometry() {
        // Every square maps back to itself through render coordinates.
        for file in 0..8u16 {
            for rank in 0..8u16 {
                let sq = Square::new(file as u8, rank as u8);
                let col = MARGIN_X + file * SQUARE_W + 1;
                let row = (7 - rank) * SQUARE_H;
                assert_eq!(square_at(area(), col, row, false), Some(sq));
            }
        }
    }

    #[test]
    fn renders_startpos_glyphs() {
        use gambito_engine::Position;
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let pos = Position::startpos();
        let mut terminal = Terminal::new(TestBackend::new(BOARD_W, BOARD_H)).unwrap();
        terminal
            .draw(|f| {
                f.render_widget(
                    BoardWidget {
                        pos: &pos,
                        flipped: false,
                        ascii: true,
                        selected: None,
                        targets: Bitboard::EMPTY,
                        last_move: None,
                        check: None,
                    },
                    f.area(),
                );
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        // White king on e1: display column for file e = 2 + 4*4 + 1, bottom rank row = 7*2.
        assert_eq!(buffer[(MARGIN_X + 4 * SQUARE_W + 1, 7 * SQUARE_H)].symbol(), "K");
        // Black queen on d8.
        assert_eq!(buffer[(MARGIN_X + 3 * SQUARE_W + 1, 0)].symbol(), "q");
        // File label row.
        assert_eq!(buffer[(MARGIN_X + 1, 8 * SQUARE_H)].symbol(), "a");
    }
}
