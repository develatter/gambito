use crossterm::event::{Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind};

/// Semantic input, decoupled from crossterm so screens are testable by
/// feeding Action sequences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Quit,
    Undo,
    Flip,
    FocusInput,
    ToMenu,
    Escape,
    Enter,
    Backspace,
    Up,
    Down,
    Char(char),
    Click { column: u16, row: u16 },
}

/// `text_entry` routes printable keys to Char instead of shortcuts.
pub fn map(event: &Event, text_entry: bool) -> Option<Action> {
    match event {
        Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
            KeyCode::Esc => Some(Action::Escape),
            KeyCode::Enter => Some(Action::Enter),
            KeyCode::Backspace => Some(Action::Backspace),
            KeyCode::Up => Some(Action::Up),
            KeyCode::Down => Some(Action::Down),
            KeyCode::Char(c) if text_entry => Some(Action::Char(c)),
            KeyCode::Char('q') => Some(Action::Quit),
            KeyCode::Char('u') => Some(Action::Undo),
            KeyCode::Char('f') => Some(Action::Flip),
            KeyCode::Char(':') => Some(Action::FocusInput),
            KeyCode::Char('m') => Some(Action::ToMenu),
            KeyCode::Char('k') => Some(Action::Up),
            KeyCode::Char('j') => Some(Action::Down),
            KeyCode::Char(c) => Some(Action::Char(c)),
            _ => None,
        },
        Event::Mouse(mouse) if mouse.kind == MouseEventKind::Down(MouseButton::Left) => {
            Some(Action::Click { column: mouse.column, row: mouse.row })
        }
        _ => None,
    }
}
