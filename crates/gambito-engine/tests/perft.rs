//! Perft suite: exact node counts from the Chess Programming Wiki. Any
//! movegen or apply bug shows up here as a count mismatch.

use gambito_engine::{perft, Position};

fn check(fen: &str, expected: &[u64]) {
    let pos: Position = gambito_engine::fen::parse(fen).unwrap();
    for (i, &nodes) in expected.iter().enumerate() {
        let depth = i as u32 + 1;
        assert_eq!(perft(&pos, depth), nodes, "{fen} at depth {depth}");
    }
}

#[test]
fn startpos() {
    check(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8_902, 197_281, 4_865_609],
    );
}

#[test]
fn kiwipete() {
    check(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2_039, 97_862, 4_085_603],
    );
}

#[test]
fn cpw_position_3() {
    check("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2_812, 43_238, 674_624]);
}

#[test]
fn cpw_position_4() {
    check(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9_467, 422_333],
    );
}

#[test]
fn cpw_position_5() {
    check("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1_486, 62_379, 2_103_487]);
}

#[test]
fn cpw_position_6() {
    check(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[46, 2_079, 89_890, 3_894_594],
    );
}
