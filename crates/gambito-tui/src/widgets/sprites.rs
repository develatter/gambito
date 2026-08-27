//! Multi-cell piece sprites drawn with half-block characters.
//!
//! Each terminal cell holds two vertically stacked "pixels" ('▀' colors the
//! top half via fg and the bottom half via bg), and since a cell is roughly
//! twice as tall as wide, those pixels come out square.
//!
//! One 8x8 set, 8-bit style: iconic shapes, not miniature physical pieces.
//! Scaling is integer-only (2x, 3x...) so pixels stay crisp blocks; when a
//! square can't hold 8 pixels, the board falls back to a centered glyph.

use gambito_engine::{Color as Side, Piece};
use ratatui::buffer::Buffer;
use ratatui::style::Color;

const BASE: usize = 8;

/// Indexed by `PieceKind::index()`: Pawn, Knight, Bishop, Rook, Queen, King.
/// '#' = fill, '.' = square shows through.
const SPRITES: [[&str; BASE]; 6] = [
    // Pawn: squat and big-headed.
    [
        "........",
        "........",
        "..####..",
        "..####..",
        "...##...",
        "..####..",
        ".######.",
        "........",
    ],
    // Knight: a "7" with a leg; one ear at the base of the head.
    [
        "........",
        ".....#..",
        ".######.",
        ".######.",
        "....###.",
        "...###..",
        "..####..",
        "........",
    ],
    // Bishop: a mitre with a diagonal notch.
    [
        "........",
        "...##...",
        "..####..",
        "..#.##..",
        "..##.#..",
        "..####..",
        ".######.",
        "........",
    ],
    // Rook: a block with crenellations.
    [
        "........",
        "##.##.##",
        "########",
        "########",
        ".######.",
        ".######.",
        "########",
        "........",
    ],
    // Queen: a spiked crown.
    [
        "........",
        "#..##..#",
        "#.####.#",
        "########",
        ".######.",
        ".######.",
        "########",
        "........",
    ],
    // King: a cross over a band crown.
    [
        "...##...",
        "..####..",
        "...##...",
        "..####..",
        ".######.",
        ".######.",
        "########",
        "........",
    ],
];

const WHITE_FILL: Color = Color::Rgb(255, 255, 255);
const BLACK_FILL: Color = Color::Rgb(25, 22, 20);

/// Pixels available for a sprite inside a square, leaving a little air.
fn canvas(square_w: u16, square_h: u16) -> (usize, usize) {
    (square_w.saturating_sub(1) as usize, (square_h as usize) * 2 - 2)
}

/// True when the square holds at least the 8x8 sprite at 1x; below that a
/// centered glyph is more recognizable than any pixel art.
pub fn fits(square_w: u16, square_h: u16) -> bool {
    let (pw, ph) = canvas(square_w, square_h);
    pw >= BASE && ph >= BASE
}

/// Draws `piece` centered in the square whose top-left cell is (x0, y0).
/// The square background must already be painted; empty pixels leave it be.
pub fn draw(buf: &mut Buffer, x0: u16, y0: u16, square_w: u16, square_h: u16, piece: Piece) {
    let (pw, ph) = canvas(square_w, square_h);
    let scale = (pw.min(ph)) / BASE;
    if scale == 0 {
        return;
    }
    let size = BASE * scale;

    let sprite = &SPRITES[piece.kind.index()];
    let fill = match piece.color {
        Side::White => WHITE_FILL,
        Side::Black => BLACK_FILL,
    };

    // Center the pixel box inside the square's 2-per-cell pixel rows.
    let ox = x0 + (square_w - size as u16) / 2;
    let oy_px = (square_h as usize * 2 - size) / 2;
    for py in 0..size {
        let row = sprite[py / scale].as_bytes();
        for px in 0..size {
            if row[px / scale] != b'#' {
                continue;
            }
            let cell_y = y0 + ((oy_px + py) / 2) as u16;
            let top_half = (oy_px + py) % 2 == 0;
            let cell = &mut buf[(ox + px as u16, cell_y)];
            match (cell.symbol(), top_half) {
                // Second half of a cell we already started: '▀' keeps fg on
                // top, so the bottom pixel goes to bg.
                ("▀", false) => {
                    cell.set_bg(fill);
                }
                (_, true) => {
                    cell.set_char('▀');
                    cell.set_fg(fill);
                }
                (_, false) => {
                    cell.set_char('▄');
                    cell.set_fg(fill);
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sprites_are_well_formed() {
        for sprite in &SPRITES {
            for row in sprite {
                assert_eq!(row.len(), BASE);
                assert!(row.bytes().all(|b| b == b'.' || b == b'#'));
            }
        }
    }

    #[test]
    fn fits_requires_a_full_1x_canvas() {
        assert!(fits(9, 5)); // 8x8 pixels available
        assert!(!fits(8, 5)); // only 7 wide
        assert!(!fits(9, 4)); // only 6 tall
    }

    #[test]
    fn draw_paints_half_blocks() {
        use gambito_engine::{Color as Side, PieceKind};
        use ratatui::layout::Rect;

        let mut buf = Buffer::empty(Rect::new(0, 0, 12, 6));
        draw(&mut buf, 0, 0, 12, 6, Piece::new(Side::White, PieceKind::King));
        let blocks = buf
            .content()
            .iter()
            .filter(|c| c.symbol() == "▀" || c.symbol() == "▄")
            .count();
        assert!(blocks > 10, "expected a drawn silhouette, got {blocks} block cells");
    }

    #[test]
    fn scale_is_integer_only() {
        use gambito_engine::{Color as Side, PieceKind};
        use ratatui::layout::Rect;

        // 20x9 square: canvas 19x16 -> scale 2 -> 16x16 sprite, centered.
        let mut buf = Buffer::empty(Rect::new(0, 0, 20, 9));
        draw(&mut buf, 0, 0, 20, 9, Piece::new(Side::Black, PieceKind::Rook));
        // Crenellation row (sprite row 1) is doubled: pixel rows 2-3 of 16,
        // offset (18-16)/2=1 -> pixel rows 3-4 -> cell rows 1-2 hold blocks.
        let has_blocks = |y: u16| (0..20).any(|x| buf[(x, y)].symbol() != " ");
        assert!(has_blocks(1) && has_blocks(2));
    }
}
