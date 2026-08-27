use crate::position::Position;
use crate::types::{CastlingRights, Color, Piece, Square};
use std::fmt;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FenError(pub String);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl std::error::Error for FenError {}

fn err(msg: impl Into<String>) -> FenError {
    FenError(msg.into())
}

pub fn parse(fen: &str) -> Result<Position, FenError> {
    let mut fields = fen.split_whitespace();
    let board = fields.next().ok_or_else(|| err("empty string"))?;
    let side = fields.next().unwrap_or("w");
    let castling = fields.next().unwrap_or("-");
    let en_passant = fields.next().unwrap_or("-");
    let halfmove = fields.next().unwrap_or("0");
    let fullmove = fields.next().unwrap_or("1");

    let mut pos = Position::empty();

    let mut rank = 7i8;
    let mut file = 0i8;
    for c in board.chars() {
        match c {
            '/' => {
                if file != 8 {
                    return Err(err(format!("rank {} has {} files", rank + 1, file)));
                }
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err(err("too many ranks"));
                }
            }
            '1'..='8' => file += c as i8 - b'0' as i8,
            _ => {
                let piece = Piece::from_char(c).ok_or_else(|| err(format!("bad piece '{c}'")))?;
                if file > 7 {
                    return Err(err(format!("rank {} overflows", rank + 1)));
                }
                pos.put(Square::new(file as u8, rank as u8), piece);
                file += 1;
            }
        }
    }
    if rank != 0 || file != 8 {
        return Err(err("board field is not 8x8"));
    }

    pos.side_to_move = match side {
        "w" => Color::White,
        "b" => Color::Black,
        _ => return Err(err(format!("bad side '{side}'"))),
    };

    if castling != "-" {
        for c in castling.chars() {
            let right = match c {
                'K' => CastlingRights::WHITE_KING,
                'Q' => CastlingRights::WHITE_QUEEN,
                'k' => CastlingRights::BLACK_KING,
                'q' => CastlingRights::BLACK_QUEEN,
                _ => return Err(err(format!("bad castling '{c}'"))),
            };
            pos.castling.0 |= right;
        }
    }

    if en_passant != "-" {
        let sq: Square = en_passant.parse().map_err(|_| err(format!("bad ep '{en_passant}'")))?;
        pos.en_passant = Some(sq);
    }

    pos.halfmove_clock = halfmove.parse().map_err(|_| err("bad halfmove clock"))?;
    pos.fullmove_number = fullmove.parse().map_err(|_| err("bad fullmove number"))?;

    if pos.pieces(Color::White, crate::types::PieceKind::King).count() != 1
        || pos.pieces(Color::Black, crate::types::PieceKind::King).count() != 1
    {
        return Err(err("each side needs exactly one king"));
    }

    pos.finish_hash();
    Ok(pos)
}

pub fn format(pos: &Position) -> String {
    let mut out = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match pos.piece_at(Square::new(file, rank)) {
                Some(piece) => {
                    if empty > 0 {
                        out.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    out.push(piece.to_char());
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            out.push((b'0' + empty) as char);
        }
        if rank > 0 {
            out.push('/');
        }
    }

    out.push(' ');
    out.push(if pos.side_to_move == Color::White { 'w' } else { 'b' });
    out.push(' ');
    if pos.castling == CastlingRights::NONE {
        out.push('-');
    } else {
        for (right, c) in [
            (CastlingRights::WHITE_KING, 'K'),
            (CastlingRights::WHITE_QUEEN, 'Q'),
            (CastlingRights::BLACK_KING, 'k'),
            (CastlingRights::BLACK_QUEEN, 'q'),
        ] {
            if pos.castling.has(right) {
                out.push(c);
            }
        }
    }
    out.push(' ');
    match pos.en_passant {
        Some(sq) => out.push_str(&sq.to_string()),
        None => out.push('-'),
    }
    out.push_str(&format!(" {} {}", pos.halfmove_clock, pos.fullmove_number));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_round_trip() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(format(&parse(fen).unwrap()), fen);
    }

    #[test]
    fn kiwipete_round_trip() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        assert_eq!(format(&parse(fen).unwrap()), fen);
    }

    #[test]
    fn hash_matches_recompute_after_parse() {
        let pos = parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq e3 4 12").unwrap();
        assert_eq!(pos.hash, pos.recompute_hash());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("").is_err());
        assert!(parse("8/8/8/8/8/8/8/8 w - - 0 1").is_err()); // no kings
        assert!(parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1").is_err());
    }
}
