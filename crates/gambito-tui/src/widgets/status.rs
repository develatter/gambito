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
            Color::White => "White",
            Color::Black => "Black",
        };
        let mut lines = vec![match self.game.status() {
            GameStatus::Ongoing => {
                if pos.in_check(pos.side_to_move) {
                    Line::from(format!("Check! {side} to move").red().bold())
                } else {
                    Line::from(format!("{side} to move"))
                }
            }
            GameStatus::Checkmate { winner } => {
                let winner = match winner {
                    Color::White => "White",
                    Color::Black => "Black",
                };
                Line::from(format!("Checkmate — {winner} wins").green().bold())
            }
            GameStatus::Stalemate => Line::from("Draw by stalemate".yellow()),
            GameStatus::FiftyMoveRule => Line::from("Draw: fifty-move rule".yellow()),
            GameStatus::ThreefoldRepetition => Line::from("Draw by threefold repetition".yellow()),
            GameStatus::InsufficientMaterial => Line::from("Draw: insufficient material".yellow()),
        }];
        if let Some(msg) = self.message {
            lines.push(Line::from(msg.to_string().magenta()));
        }
        Paragraph::new(lines)
            .block(Block::bordered().title(" Status "))
            .render(area, buf);
    }
}
