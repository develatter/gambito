use crate::event::Action;
use crate::screens::Transition;
use ratatui::layout::{Constraint, Flex, Layout};
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

const ITEMS: [(&str, bool); 5] = [
    ("Local game (hotseat)", true),
    ("Play the AI        (M2)", false),
    ("Online blitz        (M3)", false),
    ("Correspondence      (M4)", false),
    ("Quit", true),
];

pub struct MenuScreen {
    selected: usize,
    note: Option<&'static str>,
}

impl MenuScreen {
    pub fn new() -> MenuScreen {
        MenuScreen { selected: 0, note: None }
    }

    pub fn handle(&mut self, action: Action) -> Transition {
        match action {
            Action::Quit | Action::Escape => return Transition::Quit,
            Action::Up => {
                self.selected = self.selected.checked_sub(1).unwrap_or(ITEMS.len() - 1);
                self.note = None;
            }
            Action::Down => {
                self.selected = (self.selected + 1) % ITEMS.len();
                self.note = None;
            }
            Action::Enter => match self.selected {
                0 => return Transition::StartHotseat,
                4 => return Transition::Quit,
                _ => self.note = Some("Not available yet: coming in a later milestone."),
            },
            _ => {}
        }
        Transition::None
    }

    pub fn render(&self, frame: &mut Frame) {
        let mut lines = vec![
            Line::from("♞ g a m b i t o".bold()),
            Line::from("terminal chess".dim()),
            Line::from(""),
        ];
        for (i, (label, enabled)) in ITEMS.iter().enumerate() {
            let marker = if i == self.selected { "▸ " } else { "  " };
            let line = format!("{marker}{label}");
            lines.push(if !enabled {
                Line::from(line.dim())
            } else if i == self.selected {
                Line::from(line.bold())
            } else {
                Line::from(line)
            });
        }
        lines.push(Line::from(""));
        lines.push(Line::from(self.note.unwrap_or("↑/↓ and Enter · q to quit").dim()));

        let [area] = Layout::horizontal([Constraint::Length(44)])
            .flex(Flex::Center)
            .areas(frame.area());
        let [area] = Layout::vertical([Constraint::Length(lines.len() as u16 + 2)])
            .flex(Flex::Center)
            .areas(area);
        frame.render_widget(Paragraph::new(lines).block(Block::bordered()), area);
    }
}
