use gambito_tui::Options;
use std::process::ExitCode;

const USAGE: &str = "\
gambito — ajedrez de terminal

USO:
    gambito [OPCIONES]

OPCIONES:
    --fen <FEN>   Empieza una partida desde esta posición
    --ascii       Letras ASCII en vez de piezas Unicode
    -h, --help    Muestra esta ayuda";

fn main() -> ExitCode {
    let mut options = Options { fen: None, ascii: false };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fen" => match args.next() {
                Some(fen) => options.fen = Some(fen),
                None => {
                    eprintln!("--fen necesita un valor\n\n{USAGE}");
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
                        eprintln!("opción desconocida: {other}\n\n{USAGE}");
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
