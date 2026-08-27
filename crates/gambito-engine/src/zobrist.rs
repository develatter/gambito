//! Zobrist hashing keys, generated at compile time from a fixed seed so
//! hashes are stable across builds (they end up in the storage layer).

/// splitmix64 — tiny, well-distributed, and usable in const context.
const fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state, z ^ (z >> 31))
}

const fn key_table<const N: usize>(mut seed: u64) -> [u64; N] {
    let mut table = [0u64; N];
    let mut i = 0;
    while i < N {
        let (next, value) = splitmix64(seed);
        seed = next;
        table[i] = value;
        i += 1;
    }
    table
}

/// [color * 6 + kind][square]
pub const PIECES: [[u64; 64]; 12] = {
    let flat: [u64; 768] = key_table(0xC1E5_5EED_0000_0001);
    let mut out = [[0u64; 64]; 12];
    let mut p = 0;
    while p < 12 {
        let mut s = 0;
        while s < 64 {
            out[p][s] = flat[p * 64 + s];
            s += 1;
        }
        p += 1;
    }
    out
};

pub const SIDE_TO_MOVE: u64 = key_table::<1>(0xC1E5_5EED_0000_0002)[0];
pub const CASTLING: [u64; 16] = key_table(0xC1E5_5EED_0000_0003);
/// Indexed by en-passant file.
pub const EN_PASSANT: [u64; 8] = key_table(0xC1E5_5EED_0000_0004);
