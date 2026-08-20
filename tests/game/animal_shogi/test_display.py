"""Tests for board display."""

from shogi_ai.game.animal_shogi.board import Board
from shogi_ai.game.animal_shogi.display import board_to_str, piece_to_char
from shogi_ai.game.animal_shogi.types import PieceType, Player


def test_sente_piece_uppercase() -> None:
    assert piece_to_char(PieceType.LION, Player.SENTE) == "L"


def test_gote_piece_lowercase() -> None:
    assert piece_to_char(PieceType.LION, Player.GOTE) == "l"


def test_all_piece_chars() -> None:
    chars = {piece_to_char(pt, Player.SENTE) for pt in PieceType}
    assert chars == {"C", "G", "E", "L", "H"}


def test_initial_board_display() -> None:
    board = Board()
    output = board_to_str(board)
    lines = output.split("\n")

    assert "GOTE hand: -" in lines[0]
    assert "a b c" in lines[1]
    # Row 1: gote pieces
    assert "g l e" in lines[2]
    # Row 2: gote chick
    assert ". c ." in lines[3]
    # Row 3: sente chick
    assert ". C ." in lines[4]
    # Row 4: sente pieces
    assert "E L G" in lines[5]
    assert "SENTE hand: -" in lines[6]


def test_board_with_hand() -> None:
    board = Board(hands=((PieceType.CHICK,), (PieceType.GIRAFFE,)))
    output = board_to_str(board)
    assert "GOTE hand: G" in output
    assert "SENTE hand: C" in output
