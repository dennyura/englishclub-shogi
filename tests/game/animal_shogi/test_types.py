"""Tests for types and constants."""

from shogi_ai.game.animal_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    PIECE_MOVES,
    ROWS,
    PieceType,
    Player,
)


def test_board_dimensions() -> None:
    assert COLS == 3
    assert ROWS == 4


def test_player_opponent() -> None:
    assert Player.SENTE.opponent == Player.GOTE
    assert Player.GOTE.opponent == Player.SENTE


def test_piece_type_values() -> None:
    assert PieceType.CHICK.value == 0
    assert PieceType.HEN.value == 4
    assert len(PieceType) == 5


def test_piece_moves_defined() -> None:
    for pt in PieceType:
        assert pt in PIECE_MOVES
        assert len(PIECE_MOVES[pt]) > 0


def test_chick_moves_one_forward() -> None:
    assert PIECE_MOVES[PieceType.CHICK] == [(-1, 0)]


def test_lion_moves_all_directions() -> None:
    assert len(PIECE_MOVES[PieceType.LION]) == 8


def test_hand_piece_types() -> None:
    assert PieceType.CHICK in HAND_PIECE_TYPES
    assert PieceType.GIRAFFE in HAND_PIECE_TYPES
    assert PieceType.ELEPHANT in HAND_PIECE_TYPES
    assert PieceType.LION not in HAND_PIECE_TYPES
    assert PieceType.HEN not in HAND_PIECE_TYPES
