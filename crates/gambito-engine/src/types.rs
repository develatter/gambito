use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    pub const ALL: [PieceKind; 6] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Lowercase letter used in FEN and UCI promotion suffixes.
    pub fn to_char(self) -> char {
        match self {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        }
    }

    pub fn from_char(c: char) -> Option<PieceKind> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceKind::Pawn),
            'n' => Some(PieceKind::Knight),
            'b' => Some(PieceKind::Bishop),
            'r' => Some(PieceKind::Rook),
            'q' => Some(PieceKind::Queen),
            'k' => Some(PieceKind::King),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub const fn new(color: Color, kind: PieceKind) -> Piece {
        Piece { color, kind }
    }

    /// FEN letter: uppercase for White, lowercase for Black.
    pub fn to_char(self) -> char {
        let c = self.kind.to_char();
        match self.color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        let kind = PieceKind::from_char(c)?;
        let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
        Some(Piece { color, kind })
    }
}

/// A board square, 0 = a1 .. 63 = h8 (little-endian rank-file mapping).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Square(pub u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A8: Square = Square(56);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    #[inline]
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    /// 0 = a-file .. 7 = h-file.
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// 0 = rank 1 .. 7 = rank 8.
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Offset by (dfile, drank), returning None if it walks off the board.
    pub fn offset(self, dfile: i8, drank: i8) -> Option<Square> {
        let f = self.file() as i8 + dfile;
        let r = self.rank() as i8 + drank;
        if (0..8).contains(&f) && (0..8).contains(&r) {
            Some(Square::new(f as u8, r as u8))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", (b'a' + self.file()) as char, self.rank() + 1)
    }
}

impl FromStr for Square {
    type Err = ();

    fn from_str(s: &str) -> Result<Square, ()> {
        let mut chars = s.chars();
        let file = chars.next().ok_or(())?;
        let rank = chars.next().ok_or(())?;
        if chars.next().is_some() || !('a'..='h').contains(&file) || !('1'..='8').contains(&rank) {
            return Err(());
        }
        Ok(Square::new(file as u8 - b'a', rank as u8 - b'1'))
    }
}

/// Castling availability, one bit per right.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const WHITE_KING: u8 = 1;
    pub const WHITE_QUEEN: u8 = 2;
    pub const BLACK_KING: u8 = 4;
    pub const BLACK_QUEEN: u8 = 8;

    pub const NONE: CastlingRights = CastlingRights(0);
    pub const ALL: CastlingRights = CastlingRights(15);

    #[inline]
    pub fn has(self, right: u8) -> bool {
        self.0 & right != 0
    }

    #[inline]
    pub fn remove(&mut self, rights: u8) {
        self.0 &= !rights;
    }

    pub fn king_side(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_KING),
            Color::Black => self.has(Self::BLACK_KING),
        }
    }

    pub fn queen_side(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_QUEEN),
            Color::Black => self.has(Self::BLACK_QUEEN),
        }
    }
}
