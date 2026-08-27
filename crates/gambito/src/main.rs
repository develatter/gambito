use gambito_tui::Options;
use std::process::ExitCode;

const USAGE: &str = "\
gambito — terminal chess

USAGE:
    gambito [OPTIONS]

OPTIONS:
    --fen <FEN>   Start a game from this position
    --ascii       ASCII letters instead of Unicode pieces
    -h, --help    Show this help";

fn main() -> ExitCode {
    let mut options = Options { fen: None, ascii: false };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fen" => match args.next() {
                Some(fen) => options.fen = Some(fen),
                None => {
                    eprintln!("--fen needs a value\n\n{USAGE}");
                    return ExitCode::FAILURE;
                }
            },
            "--ascii" => options.ascii = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => {
                let value = other.strip_prefix("--fen=");
                match value {
                    Some(fen) => options.fen = Some(fen.to_string()),
                    None => {
                        eprintln!("unknown option: {other}\n\n{USAGE}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
    }

    match gambito_tui::run(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
