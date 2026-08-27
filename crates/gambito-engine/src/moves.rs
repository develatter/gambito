use crate::types::{PieceKind, Square};
use std::fmt;

/// Move flags, stored in the top 4 bits of a packed move.
///
/// The encoding follows the classic from-to-flags scheme: bit 2 marks a
/// capture, bit 3 marks a promotion, and the low two bits select the variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MoveFlags {
    Quiet = 0,
    DoublePush = 1,
    CastleKing = 2,
    CastleQueen = 3,
    Capture = 4,
    EnPassant = 5,
    PromoKnight = 8,
    PromoBishop = 9,
    PromoRook = 10,
    PromoQueen = 11,
    PromoCaptureKnight = 12,
    PromoCaptureBishop = 13,
    PromoCaptureRook = 14,
    PromoCaptureQueen = 15,
}

impl MoveFlags {
    pub const fn promo(kind: PieceKind, capture: bool) -> MoveFlags {
        match (kind, capture) {
            (PieceKind::Knight, false) => MoveFlags::PromoKnight,
            (PieceKind::Bishop, false) => MoveFlags::PromoBishop,
            (PieceKind::Rook, false) => MoveFlags::PromoRook,
            (PieceKind::Queen, false) => MoveFlags::PromoQueen,
            (PieceKind::Knight, true) => MoveFlags::PromoCaptureKnight,
            (PieceKind::Bishop, true) => MoveFlags::PromoCaptureBishop,
            (PieceKind::Rook, true) => MoveFlags::PromoCaptureRook,
            (PieceKind::Queen, true) => MoveFlags::PromoCaptureQueen,
            _ => MoveFlags::PromoQueen,
        }
    }
}

/// A move packed into 16 bits: from (6) | to (6) | flags (4).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(pub u16);

impl Move {
    #[inline]
    pub const fn new(from: Square, to: Square, flags: MoveFlags) -> Move {
        Move(from.0 as u16 | (to.0 as u16) << 6 | (flags as u16) << 12)
    }

    #[inline]
    pub const fn from(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    #[inline]
    pub const fn to(self) -> Square {
        Square((self.0 >> 6 & 63) as u8)
    }

    #[inline]
    pub fn flags(self) -> MoveFlags {
        match self.0 >> 12 {
            0 => MoveFlags::Quiet,
            1 => MoveFlags::DoublePush,
            2 => MoveFlags::CastleKing,
            3 => MoveFlags::CastleQueen,
            4 => MoveFlags::Capture,
            5 => MoveFlags::EnPassant,
            8 => MoveFlags::PromoKnight,
            9 => MoveFlags::PromoBishop,
            10 => MoveFlags::PromoRook,
            11 => MoveFlags::PromoQueen,
            12 => MoveFlags::PromoCaptureKnight,
            13 => MoveFlags::PromoCaptureBishop,
            14 => MoveFlags::PromoCaptureRook,
            15 => MoveFlags::PromoCaptureQueen,
            _ => unreachable!("invalid move flags"),
        }
    }

    #[inline]
    pub fn is_capture(self) -> bool {
        self.0 >> 12 & 4 != 0
    }

    #[inline]
    pub fn is_promotion(self) -> bool {
        self.0 >> 12 & 8 != 0
    }

    #[inline]
    pub fn is_castle(self) -> bool {
        matches!(self.flags(), MoveFlags::CastleKing | MoveFlags::CastleQueen)
    }

    pub fn promotion(self) -> Option<PieceKind> {
        match self.flags() {
            MoveFlags::PromoKnight | MoveFlags::PromoCaptureKnight => Some(PieceKind::Knight),
            MoveFlags::PromoBishop | MoveFlags::PromoCaptureBishop => Some(PieceKind::Bishop),
            MoveFlags::PromoRook | MoveFlags::PromoCaptureRook => Some(PieceKind::Rook),
            MoveFlags::PromoQueen | MoveFlags::PromoCaptureQueen => Some(PieceKind::Queen),
            _ => None,
        }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({}{}, {:?})", self.from(), self.to(), self.flags())
    }
}

/// Fixed-capacity move buffer; 256 exceeds the known maximum of 218 legal
/// moves in any reachable position.
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList { moves: [Move(0); 256], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.as_slice().iter()
    }
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
