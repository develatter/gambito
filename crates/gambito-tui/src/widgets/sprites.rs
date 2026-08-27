//! Multi-cell piece sprites drawn with half-block characters.
//!
//! Each terminal cell holds two vertically stacked "pixels" ('▀' colors the
//! top half via fg and the bottom half via bg), and since a cell is roughly
//! twice as tall as wide, those pixels come out square. Sprites are authored
//! once at 12x12 and nearest-neighbor scaled to whatever the square allows.

use gambito_engine::{Color as Side, Piece};
use ratatui::buffer::Buffer;
use ratatui::style::Color;

/// Silhouettes authored for small squares (~8-11 px), indexed by
/// `PieceKind::index()`: Pawn, Knight, Bishop, Rook, Queen, King. '#' = fill.
/// Nearest-neighbor downscaling destroys detail, so shapes are authored at
/// the resolutions they'll actually show at and only ever scaled up a little.
const SPRITES_8: [[&str; 8]; 6] = [
    // Pawn
    [
        "........",
        "...##...",
        "..####..",
        "...##...",
        "...##...",
        "..####..",
        ".######.",
        ".######.",
    ],
    // Knight
    [
        "..#.....",
        ".####...",
        "######..",
        "##.###..",
        "..####..",
        "..###...",
        ".#####..",
        "#######.",
    ],
    // Bishop
    [
        "...##...",
        "..#.##..",
        "..####..",
        "...##...",
        "..####..",
        "...##...",
        ".######.",
        ".######.",
    ],
    // Rook
    [
        ".#.##.#.",
        ".######.",
        "..####..",
        "..####..",
        "..####..",
        "..####..",
        ".######.",
        "########",
    ],
    // Queen
    [
        "#..##..#",
        "##.##.##",
        ".######.",
        "..####..",
        "..####..",
        "...##...",
        ".######.",
        "########",
    ],
    // King
    [
        "...##...",
        "..####..",
        "...##...",
        ".######.",
        "########",
        "..####..",
        "..####..",
        ".######.",
    ],
];

/// Silhouettes for large squares (12 px and up).
const BASE: usize = 12;
const SPRITES: [[&str; BASE]; 6] = [
    // Pawn
    [
        "............",
        "............",
        ".....##.....",
        "....####....",
        "....####....",
        ".....##.....",
        "....####....",
        "...######...",
        "...######...",
        "....####....",
        "..########..",
        ".##########.",
    ],
    // Knight
    [
        "............",
        "....##.#....",
        "...#####....",
        "..#######...",
        ".########...",
        ".##..####...",
        "....#####...",
        "...#####....",
        "..#####.....",
        "..######....",
        ".########...",
        ".#########..",
    ],
    // Bishop
    [
        ".....##.....",
        "....####....",
        "...##.###...",
        "...##.###...",
        "...######...",
        "....####....",
        "....####....",
        ".....##.....",
        "...######...",
        "....####....",
        "..########..",
        ".##########.",
    ],
    // Rook
    [
        "............",
        ".##.####.##.",
        ".##########.",
        ".##########.",
        "..########..",
        "...######...",
        "...######...",
        "...######...",
        "...######...",
        "..########..",
        ".##########.",
        ".##########.",
    ],
    // Queen
    [
        ".#...##...#.",
        ".#...##...#.",
        ".##..##..##.",
        ".##########.",
        "..########..",
        "...######...",
        "...######...",
        "....####....",
        "...######...",
        "....####....",
        "..########..",
        ".##########.",
    ],
    // King
    [
        ".....##.....",
        "....####....",
        ".....##.....",
        "..########..",
        ".##########.",
        ".##########.",
        "..########..",
        "...######...",
        "...######...",
        "...######...",
        "..########..",
        ".##########.",
    ],
];

const MAX_SIZE: usize = 32;

const WHITE_FILL: Color = Color::Rgb(255, 255, 255);
const WHITE_EDGE: Color = Color::Rgb(60, 50, 40);
const BLACK_FILL: Color = Color::Rgb(25, 22, 20);
const BLACK_EDGE: Color = Color::Rgb(235, 230, 220);

#[derive(Clone, Copy, PartialEq)]
enum Px {
    Empty,
    Fill,
    Edge,
}

/// Smallest square (in cells) where sprites beat a single glyph.
pub fn fits(square_w: u16, square_h: u16) -> bool {
    square_w >= 6 && square_h >= 3
}

/// Draws `piece` centered in the square whose top-left cell is (x0, y0).
/// The square background must already be painted; empty pixels leave it be.
pub fn draw(buf: &mut Buffer, x0: u16, y0: u16, square_w: u16, square_h: u16, piece: Piece) {
    let pw = square_w.saturating_sub(2) as usize;
    let ph = (square_h as usize) * 2 - 2;
    let size = pw.min(ph).min(MAX_SIZE);
    if size < 4 {
        return;
    }

    // Pick the set that needs the least scaling (never scale detail away).
    let (rows, base): (&[&str], usize) = if size >= BASE {
        (&SPRITES[piece.kind.index()], BASE)
    } else {
        (&SPRITES_8[piece.kind.index()], 8)
    };
    let mut grid = [[Px::Empty; MAX_SIZE]; MAX_SIZE];
    for py in 0..size {
        let row = rows[py * base / size].as_bytes();
        for px in 0..size {
            if row[px * base / size] == b'#' {
                grid[py][px] = Px::Fill;
            }
        }
    }
    // Outline = empty pixels touching the silhouette (4-neighborhood), so the
    // piece stays visible on the square that matches its own color.
    for py in 0..size {
        for px in 0..size {
            if grid[py][px] != Px::Empty {
                continue;
            }
            let filled = |y: isize, x: isize| {
                y >= 0
                    && x >= 0
                    && (y as usize) < size
                    && (x as usize) < size
                    && grid[y as usize][x as usize] == Px::Fill
            };
            let (y, x) = (py as isize, px as isize);
            if filled(y - 1, x) || filled(y + 1, x) || filled(y, x - 1) || filled(y, x + 1) {
                grid[py][px] = Px::Edge;
            }
        }
    }

    let (fill, edge) = match piece.color {
        Side::White => (WHITE_FILL, WHITE_EDGE),
        Side::Black => (BLACK_FILL, BLACK_EDGE),
    };
    let paint = |px: Px| if px == Px::Fill { fill } else { edge };

    // Center the pixel box inside the square's 2-per-cell pixel rows.
    let ox = x0 + (square_w - size as u16) / 2;
    let oy_px = (square_h as usize * 2 - size) / 2;
    for py in 0..size {
        for px in 0..size {
            if grid[py][px] == Px::Empty {
                continue;
            }
            let cell_y = y0 + ((oy_px + py) / 2) as u16;
            let top_half = (oy_px + py) % 2 == 0;
            let color = paint(grid[py][px]);
            let cell = &mut buf[(ox + px as u16, cell_y)];
            match (cell.symbol(), top_half) {
                // Second half of a cell we already started: '▀' keeps fg on
                // top, so the bottom pixel goes to bg.
                ("▀", false) => {
                    cell.set_bg(color);
                }
                (_, true) => {
                    cell.set_char('▀');
                    cell.set_fg(color);
                }
                (_, false) => {
                    cell.set_char('▄');
                    cell.set_fg(color);
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
        for sprite in &SPRITES_8 {
            for row in sprite {
                assert_eq!(row.len(), 8);
                assert!(row.bytes().all(|b| b == b'.' || b == b'#'));
            }
        }
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
}
