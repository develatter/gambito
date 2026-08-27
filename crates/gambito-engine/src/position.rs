use crate::attacks;
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveFlags};
use crate::types::{CastlingRights, Color, Piece, PieceKind, Square};
use crate::zobrist;

/// Full board state at one ply. Copy-make: `apply` returns a new Position.
///
/// Bitboards are the source of truth for move generation; the mailbox mirrors
/// them for O(1) "what sits here" lookups (SAN, TUI, apply).
#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    piece_bb: [[Bitboard; 6]; 2],
    color_bb: [Bitboard; 2],
    mailbox: [Option<Piece>; 64],
    pub side_to_move: Color,
    pub castling: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    pub hash: u64,
}

impl Position {
    pub fn empty() -> Position {
        Position {
            piece_bb: [[Bitboard::EMPTY; 6]; 2],
            color_bb: [Bitboard::EMPTY; 2],
            mailbox: [None; 64],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
        }
    }

    pub fn startpos() -> Position {
        crate::fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos FEN is valid")
    }

    #[inline]
    pub fn pieces(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.piece_bb[color.index()][kind.index()]
    }

    #[inline]
    pub fn occupied_by(&self, color: Color) -> Bitboard {
        self.color_bb[color.index()]
    }

    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.color_bb[0] | self.color_bb[1]
    }

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index()]
    }

    pub fn king_square(&self, color: Color) -> Square {
        self.pieces(color, PieceKind::King)
            .first()
            .expect("side has a king")
    }

    pub(crate) fn put(&mut self, sq: Square, piece: Piece) {
        self.piece_bb[piece.color.index()][piece.kind.index()].set(sq);
        self.color_bb[piece.color.index()].set(sq);
        self.mailbox[sq.index()] = Some(piece);
        self.hash ^= zobrist::PIECES[piece.color.index() * 6 + piece.kind.index()][sq.index()];
    }

    fn take(&mut self, sq: Square) -> Piece {
        let piece = self.mailbox[sq.index()].expect("take from occupied square");
        self.piece_bb[piece.color.index()][piece.kind.index()].clear(sq);
        self.color_bb[piece.color.index()].clear(sq);
        self.mailbox[sq.index()] = None;
        self.hash ^= zobrist::PIECES[piece.color.index() * 6 + piece.kind.index()][sq.index()];
        piece
    }

    /// All pieces of `by` attacking `sq` under the given occupancy.
    pub fn attackers_to(&self, sq: Square, by: Color, occupied: Bitboard) -> Bitboard {
        // Symmetry trick: knights that attack sq are on knight-attack squares
        // from sq; a pawn of `by` attacks sq iff it sits on a square that a
        // pawn of the *other* color on sq would attack.
        let queens = self.pieces(by, PieceKind::Queen);
        (attacks::knight_attacks(sq) & self.pieces(by, PieceKind::Knight))
            | (attacks::king_attacks(sq) & self.pieces(by, PieceKind::King))
            | (attacks::pawn_attacks(by.opposite(), sq) & self.pieces(by, PieceKind::Pawn))
            | (attacks::bishop_attacks(sq, occupied) & (self.pieces(by, PieceKind::Bishop) | queens))
            | (attacks::rook_attacks(sq, occupied) & (self.pieces(by, PieceKind::Rook) | queens))
    }

    pub fn is_attacked(&self, sq: Square, by: Color) -> bool {
        !self.attackers_to(sq, by, self.occupied()).is_empty()
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.is_attacked(self.king_square(color), color.opposite())
    }

    /// Applies a move assumed to be pseudo-legal for the side to move.
    pub fn apply(&self, mv: Move) -> Position {
        let mut pos = self.clone();
        let us = pos.side_to_move;
        let from = mv.from();
        let to = mv.to();
        let flags = mv.flags();

        // Clear the old en-passant file from the hash before recomputing it.
        if let Some(ep) = pos.en_passant.take() {
            pos.hash ^= zobrist::EN_PASSANT[ep.file() as usize];
        }

        let piece = pos.take(from);
        pos.halfmove_clock += 1;
        if piece.kind == PieceKind::Pawn {
            pos.halfmove_clock = 0;
        }

        match flags {
            MoveFlags::Capture
            | MoveFlags::PromoCaptureKnight
            | MoveFlags::PromoCaptureBishop
            | MoveFlags::PromoCaptureRook
            | MoveFlags::PromoCaptureQueen => {
                pos.take(to);
                pos.halfmove_clock = 0;
            }
            MoveFlags::EnPassant => {
                let victim = Square::new(to.file(), from.rank());
                pos.take(victim);
                pos.halfmove_clock = 0;
            }
            _ => {}
        }

        match flags {
            MoveFlags::CastleKing | MoveFlags::CastleQueen => {
                pos.put(to, piece);
                let rank = from.rank();
                let (rook_from, rook_to) = if flags == MoveFlags::CastleKing {
                    (Square::new(7, rank), Square::new(5, rank))
                } else {
                    (Square::new(0, rank), Square::new(3, rank))
                };
                let rook = pos.take(rook_from);
                pos.put(rook_to, rook);
            }
            _ => {
                let placed = match mv.promotion() {
                    Some(kind) => Piece::new(us, kind),
                    None => piece,
                };
                pos.put(to, placed);
            }
        }

        if flags == MoveFlags::DoublePush {
            let ep = Square::new(from.file(), (from.rank() + to.rank()) / 2);
            pos.en_passant = Some(ep);
            pos.hash ^= zobrist::EN_PASSANT[ep.file() as usize];
        }

        let old_castling = pos.castling;
        pos.castling.remove(castling_touched(from) | castling_touched(to));
        if pos.castling != old_castling {
            pos.hash ^= zobrist::CASTLING[old_castling.0 as usize];
            pos.hash ^= zobrist::CASTLING[pos.castling.0 as usize];
        }

        if us == Color::Black {
            pos.fullmove_number += 1;
        }
        pos.side_to_move = us.opposite();
        pos.hash ^= zobrist::SIDE_TO_MOVE;
        pos
    }

    /// Recomputes the hash from scratch; test oracle for incremental updates.
    pub fn recompute_hash(&self) -> u64 {
        let mut hash = 0u64;
        for sq in 0..64u8 {
            if let Some(piece) = self.mailbox[sq as usize] {
                hash ^= zobrist::PIECES[piece.color.index() * 6 + piece.kind.index()][sq as usize];
            }
        }
        if self.side_to_move == Color::Black {
            hash ^= zobrist::SIDE_TO_MOVE;
        }
        hash ^= zobrist::CASTLING[self.castling.0 as usize];
        if let Some(ep) = self.en_passant {
            hash ^= zobrist::EN_PASSANT[ep.file() as usize];
        }
        hash
    }

    /// Called once after building a position (FEN parse) to fold in the
    /// non-piece state that `put` alone doesn't cover.
    pub(crate) fn finish_hash(&mut self) {
        if self.side_to_move == Color::Black {
            self.hash ^= zobrist::SIDE_TO_MOVE;
        }
        self.hash ^= zobrist::CASTLING[self.castling.0 as usize];
        if let Some(ep) = self.en_passant {
            self.hash ^= zobrist::EN_PASSANT[ep.file() as usize];
        }
    }
}

/// Castling rights lost when a move touches this square (moves *or* captures).
fn castling_touched(sq: Square) -> u8 {
    match sq {
        Square::E1 => CastlingRights::WHITE_KING | CastlingRights::WHITE_QUEEN,
        Square::A1 => CastlingRights::WHITE_QUEEN,
        Square::H1 => CastlingRights::WHITE_KING,
        Square::E8 => CastlingRights::BLACK_KING | CastlingRights::BLACK_QUEEN,
        Square::A8 => CastlingRights::BLACK_QUEEN,
        Square::H8 => CastlingRights::BLACK_KING,
        _ => 0,
    }
}
