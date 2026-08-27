//! Ratatui front-end for gambito: menu + hotseat game screen with mouse
//! and SAN keyboard input.

mod app;
mod event;
mod screens;
mod widgets;

pub use app::{run, Options};

/// Renders a position at the given size and returns the raw glyphs, one row
/// per line. Dev aid for eyeballing sprite silhouettes (see examples/).
pub fn debug_board_dump(pos: &gambito_engine::Position, width: u16, height: u16) -> String {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal
        .draw(|f| {
            let geom = widgets::board::BoardGeometry::fit(f.area());
            f.render_widget(
                widgets::board::BoardWidget {
                    pos,
                    geom,
                    flipped: false,
                    ascii: false,
                    selected: None,
                    targets: gambito_engine::Bitboard::EMPTY,
                    last_move: None,
                    check: None,
                },
                f.area(),
            );
        })
        .unwrap();
    // Decode each cell into its two half-block pixels so silhouettes are
    // visible without color: '#' white fill, '@' black fill, '+' outline.
    let px_char = |color: ratatui::style::Color| match color {
        ratatui::style::Color::Rgb(255, 255, 255) => '#',
        ratatui::style::Color::Rgb(25, 22, 20) => '@',
        ratatui::style::Color::Rgb(60, 50, 40) | ratatui::style::Color::Rgb(235, 230, 220) => '+',
        _ => '.',
    };
    let buffer = terminal.backend().buffer();
    let mut out = String::new();
    for y in 0..height {
        let mut top_row = String::new();
        let mut bottom_row = String::new();
        for x in 0..width {
            let cell = &buffer[(x, y)];
            let (top, bottom) = match cell.symbol() {
                "▀" => (cell.fg, cell.bg),
                "▄" => (cell.bg, cell.fg),
                _ => (cell.bg, cell.bg),
            };
            top_row.push(px_char(top));
            bottom_row.push(px_char(bottom));
        }
        out.push_str(&top_row);
        out.push('\n');
        out.push_str(&bottom_row);
        out.push('\n');
    }
    out
}
