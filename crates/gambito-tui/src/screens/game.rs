use crate::event::Action;
use crate::screens::Transition;
use crate::widgets::board::{BoardGeometry, BoardWidget};
use crate::widgets::movelist::MoveListWidget;
use crate::widgets::promo::PromoPopup;
use crate::widgets::saninput::InputBar;
use crate::widgets::status::StatusWidget;
use gambito_engine::{Bitboard, Game, Move, PieceKind, Square};
use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::Block;
use ratatui::Frame;

pub struct GameScreen {
    game: Game,
    ascii: bool,
    flipped: bool,
    selected: Option<Square>,
    /// Some while the SAN prompt has focus.
    input: Option<String>,
    message: Option<String>,
    /// Pending promotion: origin and target chosen, piece kind not yet.
    promo: Option<(Square, Square)>,
    confirm_quit: bool,
    /// Board geometry from the last render, for mouse hit-testing.
    geom: BoardGeometry,
}

impl GameScreen {
    pub fn new(game: Game, ascii: bool) -> GameScreen {
        GameScreen {
            game,
            ascii,
            flipped: false,
            selected: None,
            input: None,
            message: None,
            promo: None,
            confirm_quit: false,
            geom: BoardGeometry::default(),
        }
    }

    pub fn text_entry(&self) -> bool {
        self.input.is_some()
    }

    pub fn handle(&mut self, action: Action) -> Transition {
        if self.confirm_quit {
            match action {
                Action::Enter | Action::Char('y') | Action::Char('s') | Action::Quit => {
                    return Transition::Quit;
                }
                _ => {
                    self.confirm_quit = false;
                    self.message = None;
                    return Transition::None;
                }
            }
        }
        if let Some((from, to)) = self.promo {
            // In the popup, `q` means queen, not quit.
            let kind = match action {
                Action::Quit => Some(PieceKind::Queen),
                Action::Char('r') => Some(PieceKind::Rook),
                Action::Char('b') => Some(PieceKind::Bishop),
                Action::Char('n') => Some(PieceKind::Knight),
                Action::Escape => {
                    self.promo = None;
                    return Transition::None;
                }
                _ => None,
            };
            if let Some(kind) = kind {
                self.promo = None;
                if let Some(mv) = self.find_move(from, to, Some(kind)) {
                    self.play(mv);
                }
            }
            return Transition::None;
        }
        if self.input.is_some() {
            match action {
                Action::Escape => self.input = None,
                Action::Enter => self.submit_san(),
                Action::Backspace => {
                    self.input.as_mut().map(String::pop);
                }
                Action::Char(c) => self.input.as_mut().unwrap().push(c),
                Action::Click { column, row } => self.click(column, row),
                _ => {}
            }
            return Transition::None;
        }
        match action {
            Action::Quit => {
                self.confirm_quit = true;
                self.message = Some("Quit gambito? Enter/y confirms, any other key cancels.".into());
            }
            Action::ToMenu => return Transition::ToMenu,
            Action::Undo => {
                if self.game.undo() {
                    self.selected = None;
                    self.message = None;
                } else {
                    self.message = Some("Nothing to undo.".into());
                }
            }
            Action::Flip => self.flipped = !self.flipped,
            Action::FocusInput => {
                self.input = Some(String::new());
                self.selected = None;
            }
            Action::Escape => {
                if self.selected.is_some() {
                    self.selected = None;
                } else {
                    return Transition::ToMenu;
                }
            }
            Action::Click { column, row } => self.click(column, row),
            _ => {}
        }
        Transition::None
    }

    fn click(&mut self, column: u16, row: u16) {
        let Some(sq) = self.geom.square_at(column, row, self.flipped) else {
            return;
        };
        if self.game.status().is_over() {
            return;
        }
        let pos = self.game.position();
        let own = pos.piece_at(sq).is_some_and(|p| p.color == pos.side_to_move);
        if let Some(from) = self.selected {
            if self.find_move(from, sq, Some(PieceKind::Queen)).is_some() {
                self.selected = None;
                self.promo = Some((from, sq));
                return;
            }
            if let Some(mv) = self.find_move(from, sq, None) {
                self.selected = None;
                self.play(mv);
                return;
            }
        }
        self.selected = if own { Some(sq) } else { None };
    }

    /// Looks up the legal move from→to; `promo` narrows promotion variants
    /// (None matches only non-promotions).
    fn find_move(&self, from: Square, to: Square, promo: Option<PieceKind>) -> Option<Move> {
        self.game
            .legal_moves()
            .iter()
            .copied()
            .find(|mv| mv.from() == from && mv.to() == to && mv.promotion() == promo)
    }

    fn play(&mut self, mv: Move) {
        self.game.play(mv);
        self.message = None;
        self.input = None;
    }

    fn submit_san(&mut self) {
        let text = self.input.clone().unwrap_or_default();
        if text.trim().is_empty() {
            self.input = None;
            return;
        }
        if self.game.status().is_over() {
            self.message = Some("The game is over.".into());
            self.input = None;
            return;
        }
        match self.game.play_san(&text) {
            Some(_) => {
                self.selected = None;
                self.message = None;
                self.input = None;
            }
            None => self.message = Some(format!("Invalid move: {text}")),
        }
    }

    /// Squares the currently selected piece can move to (for highlighting).
    fn targets(&self) -> Bitboard {
        let Some(from) = self.selected else {
            return Bitboard::EMPTY;
        };
        let mut targets = Bitboard::EMPTY;
        for &mv in &self.game.legal_moves() {
            if mv.from() == from {
                targets.set(mv.to());
            }
        }
        targets
    }

    pub fn render(&mut self, frame: &mut Frame) {
        // The board owns the flexible space; the side column is fixed and
        // deliberately understated.
        let [main, input_bar] =
            Layout::vertical([Constraint::Min(10), Constraint::Length(1)]).areas(frame.area());
        let [board_col, side_col] =
            Layout::horizontal([Constraint::Min(40), Constraint::Length(28)]).areas(main);
        let [movelist_area, status_area] =
            Layout::vertical([Constraint::Min(6), Constraint::Length(4)]).areas(side_col);

        let board_block = Block::bordered().title(" gambito ");
        let inner = board_block.inner(board_col);
        self.geom = BoardGeometry::fit(inner);
        frame.render_widget(board_block, board_col);

        let pos = self.game.position();
        let check = pos
            .in_check(pos.side_to_move)
            .then(|| pos.king_square(pos.side_to_move));
        let last_move = self.game.moves_played().last().map(|p| (p.mv.from(), p.mv.to()));
        frame.render_widget(
            BoardWidget {
                pos,
                geom: self.geom,
                flipped: self.flipped,
                ascii: self.ascii,
                selected: self.selected,
                targets: self.targets(),
                last_move,
                check,
            },
            inner,
        );

        frame.render_widget(MoveListWidget { moves: self.game.moves_played() }, movelist_area);
        frame.render_widget(
            StatusWidget { game: &self.game, message: self.message.as_deref() },
            status_area,
        );
        frame.render_widget(InputBar { input: self.input.as_deref() }, input_bar);

        if self.promo.is_some() {
            frame.render_widget(PromoPopup, frame.area());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::layout::Rect;

    fn test_geom() -> BoardGeometry {
        BoardGeometry { area: Rect::new(0, 0, 34, 17), square_w: 4, square_h: 2 }
    }

    fn screen() -> GameScreen {
        let mut s = GameScreen::new(Game::new(), true);
        // Fixed geometry so clicks can be computed: board at origin.
        s.geom = test_geom();
        s
    }

    fn click_square(s: &mut GameScreen, name: &str) {
        let sq: Square = name.parse().unwrap();
        let column = 2 + sq.file() as u16 * 4 + 1;
        let row = (7 - sq.rank() as u16) * 2;
        s.handle(Action::Click { column, row });
    }

    #[test]
    fn click_to_move() {
        let mut s = screen();
        click_square(&mut s, "e2");
        assert_eq!(s.selected, Some("e2".parse().unwrap()));
        click_square(&mut s, "e4");
        assert_eq!(s.game.moves_played().len(), 1);
        assert_eq!(s.game.moves_played()[0].san, "e4");
    }

    #[test]
    fn san_input_flow() {
        let mut s = screen();
        s.handle(Action::FocusInput);
        for c in "Nf3".chars() {
            s.handle(Action::Char(c));
        }
        s.handle(Action::Enter);
        assert_eq!(s.game.moves_played()[0].san, "Nf3");
        assert!(!s.text_entry());
    }

    #[test]
    fn invalid_san_keeps_focus_and_reports() {
        let mut s = screen();
        s.handle(Action::FocusInput);
        s.handle(Action::Char('z'));
        s.handle(Action::Enter);
        assert!(s.message.is_some());
        assert!(s.text_entry());
    }

    #[test]
    fn promotion_asks_for_piece() {
        let mut s = GameScreen::new(
            Game::from_fen("2k3r1/5P2/8/8/8/8/8/4K3 w - - 0 1").unwrap(),
            true,
        );
        s.geom = test_geom();
        click_square(&mut s, "f7");
        click_square(&mut s, "g8");
        assert!(s.promo.is_some());
        s.handle(Action::Char('n')); // underpromotes to knight
        assert_eq!(s.game.moves_played()[0].san, "fxg8=N");
    }

    #[test]
    fn render_smoke_at_many_sizes() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        for (w, h) in [(20, 8), (40, 12), (80, 24), (127, 62), (220, 70)] {
            let mut s = GameScreen::new(Game::new(), true);
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| s.render(f)).unwrap();
        }
    }

    #[test]
    fn quit_needs_confirmation() {
        let mut s = screen();
        assert!(matches!(s.handle(Action::Quit), Transition::None));
        assert!(matches!(s.handle(Action::Char('x')), Transition::None)); // cancels
        assert!(matches!(s.handle(Action::Quit), Transition::None));
        assert!(matches!(s.handle(Action::Enter), Transition::Quit));
    }
}
