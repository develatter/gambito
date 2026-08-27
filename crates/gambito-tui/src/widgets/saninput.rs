use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

/// One-line bar: SAN prompt when focused, key hints otherwise.
pub struct InputBar<'a> {
    pub input: Option<&'a str>,
}

impl Widget for InputBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let line = match self.input {
            Some(text) => Line::from(vec![
                Span::from(" jugada> ").bold(),
                Span::from(text.to_string()),
                Span::from("█").dim(),
            ]),
            None => Line::from(
                " clic: mover · : SAN · u deshacer · f girar · m menú · q salir ".dim(),
            ),
        };
        Paragraph::new(line).render(area, buf);
    }
}
