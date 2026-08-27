pub mod game;
pub mod menu;

/// What a screen asks the app to do after handling an action.
pub enum Transition {
    None,
    Quit,
    ToMenu,
    StartHotseat,
    StartVsAi,
}
