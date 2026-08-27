use gambito_engine::{Color, Game, GameStatus};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Widget};

pub struct StatusWidget<'a> {
    pub game: &'a Game,
    pub message: Option<&'a str>,
}

impl Widget for StatusWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let pos = self.game.position();
        let side = match pos.side_to_move {
            Color::White => "blancas",
            Color::Black => "negras",
        };
        let mut lines = vec![match self.game.status() {
            GameStatus::Ongoing => {
                if pos.in_check(pos.side_to_move) {
                    Line::from(format!("¡Jaque! Mueven {side}").red().bold())
                } else {
                    Line::from(format!("Mueven {side}"))
                }
            }
            GameStatus::Checkmate { winner } => {
                let winner = match winner {
                    Color::White => "blancas",
                    Color::Black => "negras",
                };
                Line::from(format!("Jaque mate — ganan {winner}").green().bold())
            }
            GameStatus::Stalemate => Line::from("Tablas por ahogado".yellow()),
            GameStatus::FiftyMoveRule => Line::from("Tablas: regla de 50 movimientos".yellow()),
            GameStatus::ThreefoldRepetition => Line::from("Tablas por triple repetición".yellow()),
            GameStatus::InsufficientMaterial => Line::from("Tablas: material insuficiente".yellow()),
        }];
        if let Some(msg) = self.message {
            lines.push(Line::from(msg.to_string().magenta()));
        }
        Paragraph::new(lines)
            .block(Block::bordered().title(" Estado "))
            .render(area, buf);
    }
}
