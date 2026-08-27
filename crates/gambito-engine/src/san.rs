//! Standard Algebraic Notation. Formatting builds the string from the move
//! and position; parsing matches the input against each legal move's SAN,
//! which sidesteps the whole disambiguation-grammar swamp.

use crate::movegen::legal_moves;
use crate::moves::{Move, MoveFlags};
use crate::position::Position;
use crate::types::PieceKind;

pub fn format(pos: &Position, mv: Move) -> String {
    let mut san = match mv.flags() {
        MoveFlags::CastleKing => "O-O".to_string(),
        MoveFlags::CastleQueen => "O-O-O".to_string(),
        _ => {
            let piece = pos.piece_at(mv.from()).expect("move starts on a piece");
            let mut s = String::new();
            if piece.kind == PieceKind::Pawn {
                if mv.is_capture() {
                    s.push((b'a' + mv.from().file()) as char);
                }
            } else {
                s.push(piece.kind.to_char().to_ascii_uppercase());
                s.push_str(&disambiguate(pos, mv, piece.kind));
            }
            if mv.is_capture() {
                s.push('x');
            }
            s.push_str(&mv.to().to_string());
            if let Some(kind) = mv.promotion() {
                s.push('=');
                s.push(kind.to_char().to_ascii_uppercase());
            }
            s
        }
    };

    let next = pos.apply(mv);
    if next.in_check(next.side_to_move) {
        san.push(if legal_moves(&next).is_empty() { '#' } else { '+' });
    }
    san
}

/// File, rank, or both — the minimal tag that makes the origin unique among
/// same-kind pieces that can also reach the target square.
fn disambiguate(pos: &Position, mv: Move, kind: PieceKind) -> String {
    let mut same_file = false;
    let mut same_rank = false;
    let mut any = false;
    for &other in &legal_moves(pos) {
        if other == mv || other.to() != mv.to() {
            continue;
        }
        let Some(p) = pos.piece_at(other.from()) else { continue };
        if p.kind != kind {
            continue;
        }
        any = true;
        if other.from().file() == mv.from().file() {
            same_file = true;
        }
        if other.from().rank() == mv.from().rank() {
            same_rank = true;
        }
    }
    if !any {
        String::new()
    } else if !same_file {
        ((b'a' + mv.from().file()) as char).to_string()
    } else if !same_rank {
        ((b'1' + mv.from().rank()) as char).to_string()
    } else {
        mv.from().to_string()
    }
}

/// Accepts SAN with or without the trailing +/#; "0-0" is normalized to "O-O".
pub fn parse(pos: &Position, input: &str) -> Option<Move> {
    let wanted = normalize(input);
    legal_moves(pos)
        .iter()
        .copied()
        .find(|&mv| normalize(&format(pos, mv)) == wanted)
}

fn normalize(s: &str) -> String {
    s.trim()
        .trim_end_matches(['+', '#', '!', '?'])
        .replace('0', "O")
}

#[cfg(test)]
mod tests {
    use crate::fen;

    fn roundtrip(fen: &str, san: &str) {
        let pos = fen::parse(fen).unwrap();
        let mv = super::parse(&pos, san).unwrap_or_else(|| panic!("'{san}' not found in {fen}"));
        assert_eq!(super::format(&pos, mv), san);
    }

    #[test]
    fn pawn_and_piece_moves() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        roundtrip(start, "e4");
        roundtrip(start, "Nf3");
    }

    #[test]
    fn file_disambiguation() {
        // Two knights on b1 and f3 can both reach d2.
        roundtrip("rnbqkb1r/pppppppp/8/8/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1", "Nbd2");
    }

    #[test]
    fn rank_disambiguation() {
        // Rooks on a1 and a5: same file, so ranks disambiguate.
        roundtrip("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "R5a3");
    }

    #[test]
    fn full_square_disambiguation() {
        // Queens on d1, d5, h1 all reach h5 -> needs full origin square
        // (and h5 eyes e8, so it lands with check).
        roundtrip("4k3/8/8/3Q4/8/8/8/3QK2Q w - - 0 1", "Qd1h5+");
    }

    #[test]
    fn en_passant_is_a_pawn_capture() {
        roundtrip("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2", "dxe6");
    }

    #[test]
    fn underpromotion_with_capture() {
        roundtrip("2k3r1/5P2/8/8/8/8/8/4K3 w - - 0 1", "fxg8=N");
    }

    #[test]
    fn castle_mate_suffix() {
        // Back-rank: O-O-O delivers mate with the rook landing on d1... build
        // a simple mate-in-one instead: Ra8#.
        let pos = crate::fen::parse("4k3/8/4K3/8/8/8/8/R7 w - - 0 1").unwrap();
        let mv = super::parse(&pos, "Ra8#").unwrap();
        assert_eq!(super::format(&pos, mv), "Ra8#");
    }

    #[test]
    fn check_suffix() {
        roundtrip("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "Ra8+");
    }
}
