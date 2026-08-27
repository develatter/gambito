use gambito_engine::PlayedMove;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Widget};

pub struct MoveListWidget<'a> {
    pub moves: &'a [PlayedMove],
}

impl Widget for MoveListWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines: Vec<Line> = Vec::new();
        for (i, pair) in self.moves.chunks(2).enumerate() {
            let white = pair[0].san.as_str();
            let black = pair.get(1).map(|m| m.san.as_str()).unwrap_or("");
            lines.push(Line::from(format!("{:>3}. {:<8} {}", i + 1, white, black)));
        }
        let block = Block::bordered().title(" Jugadas ");
        // Keep the tail visible once the list outgrows the box.
        let visible = block.inner(area).height as usize;
        let skip = lines.len().saturating_sub(visible);
        Paragraph::new(lines.split_off(skip.min(lines.len())))
            .style(Style::new().dim())
            .block(block)
            .render(area, buf);
    }
}
