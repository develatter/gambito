use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, Paragraph, Widget};

/// Centered popup asking which piece a pawn promotes to.
pub struct PromoPopup;

impl Widget for PromoPopup {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = 34.min(area.width);
        let height = 4.min(area.height);
        let popup = Rect::new(
            area.x + (area.width.saturating_sub(width)) / 2,
            area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        );
        Clear.render(popup, buf);
        Paragraph::new(vec![
            Line::from("q dama   r torre".bold()),
            Line::from("b alfil  n caballo".bold()),
        ])
        .block(Block::bordered().title(" Promoción "))
        .render(popup, buf);
    }
}
