use crate::event;
use crate::screens::game::GameScreen;
use crate::screens::menu::MenuScreen;
use crate::screens::Transition;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use gambito_ai::{MctsBrain, NnEval};
use gambito_engine::Game;
use ratatui::DefaultTerminal;
use std::io;
use std::time::Duration;

pub struct Options {
    /// Start the game screen directly from this position.
    pub fen: Option<String>,
    /// ASCII piece letters instead of Unicode glyphs.
    pub ascii: bool,
}

pub fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    // Validate the FEN before touching the terminal so errors print normally.
    let initial_game = match &options.fen {
        Some(fen) => Some(Game::from_fen(fen)?),
        None => None,
    };

    let mut terminal = ratatui::init(); // installs a panic hook that restores
    execute!(io::stdout(), EnableMouseCapture)?;
    let result = App::new(initial_game, options.ascii).run(&mut terminal);
    execute!(io::stdout(), DisableMouseCapture).ok();
    ratatui::restore();
    Ok(result?)
}

enum Screen {
    Menu(MenuScreen),
    Game(GameScreen),
}

struct App {
    screen: Screen,
    ascii: bool,
}

impl App {
    fn new(initial_game: Option<Game>, ascii: bool) -> App {
        let screen = match initial_game {
            Some(game) => Screen::Game(GameScreen::new(game, ascii)),
            None => Screen::Menu(MenuScreen::new()),
        };
        App { screen, ascii }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        loop {
            terminal.draw(|frame| match &mut self.screen {
                Screen::Menu(menu) => menu.render(frame),
                Screen::Game(game) => game.render(frame),
            })?;

            if !crossterm::event::poll(Duration::from_millis(50))? {
                continue;
            }
            let raw = crossterm::event::read()?;
            let text_entry = match &self.screen {
                Screen::Game(game) => game.text_entry(),
                Screen::Menu(_) => false,
            };
            let Some(action) = event::map(&raw, text_entry) else {
                continue;
            };
            let transition = match &mut self.screen {
                Screen::Menu(menu) => menu.handle(action),
                Screen::Game(game) => game.handle(action),
            };
            match transition {
                Transition::None => {}
                Transition::Quit => return Ok(()),
                Transition::ToMenu => self.screen = Screen::Menu(MenuScreen::new()),
                Transition::StartHotseat => {
                    self.screen = Screen::Game(GameScreen::new(Game::new(), self.ascii));
                }
                Transition::StartVsAi => {
                    let brain = Box::new(MctsBrain::new(NnEval::embedded(), 400));
                    self.screen =
                        Screen::Game(GameScreen::new(Game::new(), self.ascii).with_opponent(brain));
                }
            }
        }
    }
}
