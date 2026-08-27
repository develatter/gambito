//! Board rendering and mouse hit-testing. Both go through BoardGeometry so
//! the renderer and the mouse math can never disagree.

use crate::widgets::sprites;
use gambito_engine::{Bitboard, Color as Side, PieceKind, Position, Square};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// Rank labels to the left of the squares.
const MARGIN_X: u16 = 2;
/// File labels under the squares.
const MARGIN_Y: u16 = 1;

const LIGHT_SQ: Color = Color::Rgb(240, 217, 181);
const DARK_SQ: Color = Color::Rgb(181, 136, 99);
const SELECTED: Color = Color::Rgb(130, 151, 105);
const LAST_MOVE_LIGHT: Color = Color::Rgb(205, 210, 106);
const LAST_MOVE_DARK: Color = Color::Rgb(170, 162, 58);
const CHECK: Color = Color::Rgb(212, 83, 71);
const WHITE_PIECE: Color = Color::Rgb(255, 255, 255);
const BLACK_PIECE: Color = Color::Rgb(20, 18, 16);
const LABEL: Color = Color::Rgb(140, 140, 140);

/// Where the board sits and how big each square is, recomputed every render
/// from the space available, so the board scales with the terminal.
#[derive(Clone, Copy, Default)]
pub struct BoardGeometry {
    pub area: Rect,
    pub square_w: u16,
    pub square_h: u16,
}

impl BoardGeometry {
    /// Largest board that fits in `avail`, centered, with squares near the
    /// 2:1 cell ratio that looks square in a terminal.
    pub fn fit(avail: Rect) -> BoardGeometry {
        let usable_w = avail.width.saturating_sub(MARGIN_X);
        let usable_h = avail.height.saturating_sub(MARGIN_Y);
        let square_h = (usable_h / 8).max(1);
        let square_w = (usable_w / 8).min(square_h * 2).max(2);
        // If width was the limit, shrink height back to keep the ratio.
        let square_h = (square_w / 2).max(1);
        let w = MARGIN_X + 8 * square_w;
        let h = 8 * square_h + MARGIN_Y;
        BoardGeometry {
            area: Rect::new(
                avail.x + avail.width.saturating_sub(w) / 2,
                avail.y + avail.height.saturating_sub(h) / 2,
                w.min(avail.width),
                h.min(avail.height),
            ),
            square_w,
            square_h,
        }
    }

    /// Top-left cell of a square by display (post-flip) coordinates.
    fn origin(&self, file_disp: u16, rank_disp: u16) -> (u16, u16) {
        (
            self.area.x + MARGIN_X + file_disp * self.square_w,
            self.area.y + rank_disp * self.square_h,
        )
    }

    /// Center cell of a square, where glyphs and markers go. Even sizes
    /// round toward the lower-right, which reads as centered for glyphs
    /// that occupy the upper-left of their own cell.
    fn center(&self, file_disp: u16, rank_disp: u16) -> (u16, u16) {
        let (x, y) = self.origin(file_disp, rank_disp);
        (x + self.square_w / 2, y + self.square_h / 2)
    }

    /// Maps a terminal cell back to a board square.
    pub fn square_at(&self, column: u16, row: u16, flipped: bool) -> Option<Square> {
        let bx = self.area.x + MARGIN_X;
        if column < bx || row < self.area.y {
            return None;
        }
        let file_disp = (column - bx) / self.square_w;
        let rank_disp = (row - self.area.y) / self.square_h;
        if file_disp > 7 || rank_disp > 7 || row >= self.area.y + 8 * self.square_h {
            return None;
        }
        let file = if flipped { 7 - file_disp } else { file_disp };
        let rank = if flipped { rank_disp } else { 7 - rank_disp };
        Some(Square::new(file as u8, rank as u8))
    }
}

pub struct BoardWidget<'a> {
    pub pos: &'a Position,
    pub geom: BoardGeometry,
    pub flipped: bool,
    pub ascii: bool,
    pub selected: Option<Square>,
    pub targets: Bitboard,
    pub last_move: Option<(Square, Square)>,
    pub check: Option<Square>,
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
        let geom = self.geom;
        // A geometry the area can't hold would index outside the buffer.
        if geom.area.width < MARGIN_X + 8 * geom.square_w
            || geom.area.height < 8 * geom.square_h + MARGIN_Y
            || geom.area.right() > area.right()
            || geom.area.bottom() > area.bottom()
        {
            return;
        }
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

                let (x0, y0) = geom.origin(file_disp, rank_disp);
                for dy in 0..geom.square_h {
                    for dx in 0..geom.square_w {
                        let cell = &mut buf[(x0 + dx, y0 + dy)];
                        cell.set_char(' ');
                        cell.set_bg(bg);
                    }
                }

                let (cx, cy) = geom.center(file_disp, rank_disp);
                if let Some(piece) = self.pos.piece_at(sq) {
                    if !self.ascii && sprites::fits(geom.square_w, geom.square_h) {
                        sprites::draw(buf, x0, y0, geom.square_w, geom.square_h, piece);
                    } else {
                        let fg =
                            if piece.color == Side::White { WHITE_PIECE } else { BLACK_PIECE };
                        let cell = &mut buf[(cx, cy)];
                        cell.set_char(glyph(piece.kind, self.ascii, piece.color));
                        cell.set_fg(fg);
                    }
                    // Capture targets keep the piece and get a corner marker.
                    if self.targets.contains(sq) {
                        let cell = &mut buf[(x0 + geom.square_w - 1, y0)];
                        cell.set_char('x');
                        cell.set_fg(CHECK);
                    }
                } else if self.targets.contains(sq) {
                    let cell = &mut buf[(cx, cy)];
                    cell.set_char('•');
                    cell.set_fg(if dark { LIGHT_SQ } else { DARK_SQ });
                }
            }
            let rank_label = if self.flipped { rank_disp + 1 } else { 8 - rank_disp };
            let (_, cy) = geom.center(0, rank_disp);
            let cell = &mut buf[(geom.area.x, cy)];
            cell.set_char((b'0' + rank_label as u8) as char);
            cell.set_fg(LABEL);
        }
        for file_disp in 0..8u16 {
            let file = if self.flipped { 7 - file_disp } else { file_disp };
            let (cx, _) = geom.center(file_disp, 0);
            let cell = &mut buf[(cx, geom.area.y + 8 * geom.square_h)];
            cell.set_char((b'a' + file as u8) as char);
            cell.set_fg(LABEL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometry every earlier test assumed: 4x2 squares at the origin.
    fn small() -> BoardGeometry {
        BoardGeometry { area: Rect::new(0, 0, 34, 17), square_w: 4, square_h: 2 }
    }

    #[test]
    fn fit_scales_to_available_space() {
        // Plenty of room: height-bound, 2:1 squares, centered horizontally.
        let g = BoardGeometry::fit(Rect::new(0, 0, 120, 49));
        assert_eq!((g.square_w, g.square_h), (12, 6));
        assert_eq!(g.area.width, 2 + 8 * 12);
        assert_eq!(g.area.x, (120 - 98) / 2);
        // Width-bound: height shrinks to keep the ratio.
        let g = BoardGeometry::fit(Rect::new(0, 0, 50, 60));
        assert_eq!((g.square_w, g.square_h), (6, 3));
        // Tiny terminal still yields a usable minimum.
        let g = BoardGeometry::fit(Rect::new(0, 0, 10, 5));
        assert_eq!((g.square_w, g.square_h), (2, 1));
    }

    #[test]
    fn hit_test_corners() {
        let g = small();
        // Top-left square is a8 in normal orientation, h1 flipped.
        assert_eq!(g.square_at(2, 0, false), Some("a8".parse().unwrap()));
        assert_eq!(g.square_at(2, 0, true), Some("h1".parse().unwrap()));
        assert_eq!(g.square_at(2 + 32 - 1, 16 - 1, false), Some("h1".parse().unwrap()));
        // Label row and left margin miss.
        assert_eq!(g.square_at(2, 16, false), None);
        assert_eq!(g.square_at(0, 0, false), None);
    }

    #[test]
    fn hit_test_inverts_render_geometry() {
        for g in [small(), BoardGeometry::fit(Rect::new(3, 5, 100, 40))] {
            for file_disp in 0..8u16 {
                for rank_disp in 0..8u16 {
                    let (cx, cy) = g.center(file_disp, rank_disp);
                    let expected =
                        Square::new(file_disp as u8, 7 - rank_disp as u8);
                    assert_eq!(g.square_at(cx, cy, false), Some(expected));
                }
            }
        }
    }

    #[test]
    fn renders_startpos_glyphs_centered() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let pos = Position::startpos();
        let geom = small();
        let mut terminal = Terminal::new(TestBackend::new(34, 17)).unwrap();
        terminal
            .draw(|f| {
                f.render_widget(
                    BoardWidget {
                        pos: &pos,
                        geom,
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
        // White king on e1 sits at the center cell of its square.
        let (cx, cy) = geom.center(4, 7);
        assert_eq!(buffer[(cx, cy)].symbol(), "K");
        // Black queen on d8.
        let (cx, cy) = geom.center(3, 0);
        assert_eq!(buffer[(cx, cy)].symbol(), "q");
        // File label centered under the a-file.
        let (cx, _) = geom.center(0, 0);
        assert_eq!(buffer[(cx, 16)].symbol(), "a");
    }
}
