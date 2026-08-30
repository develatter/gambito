import io

import chess

from gambito_train.dataset import iter_samples

GOOD_GAME = """\
[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2400"]
[BlackElo "2310"]
[TimeControl "300+0"]

1. e4 e5 2. Nf3 1-0
"""

BULLET_GAME = GOOD_GAME.replace('[TimeControl "300+0"]', '[TimeControl "60+0"]')
WEAK_GAME = GOOD_GAME.replace('[WhiteElo "2400"]', '[WhiteElo "900"]')


def test_samples_carry_pov_outcome():
    samples = list(iter_samples(io.StringIO(GOOD_GAME)))
    assert len(samples) == 3
    fen0, uci0, z0 = samples[0]
    assert fen0 == chess.STARTING_FEN
    assert uci0 == "e2e4"
    assert z0 == 1  # White won and White is to move
    _, uci1, z1 = samples[1]
    assert uci1 == "e7e5"
    assert z1 == -1  # same game seen from the loser's seat


def test_filters_drop_bullet_and_weak_games():
    assert list(iter_samples(io.StringIO(BULLET_GAME))) == []
    assert list(iter_samples(io.StringIO(WEAK_GAME))) == []
