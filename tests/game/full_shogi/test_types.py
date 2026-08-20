"""Tests for full shogi types."""

from __future__ import annotations

from shogi_ai.game.full_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    PROMOTION_MAP,
    ROWS,
    UNPROMOTION_MAP,
    PieceType,
    Player,
)


def test_board_dimensions() -> None:
    assert ROWS == 9
    assert COLS == 9


def test_player_opponent() -> None:
    assert Player.SENTE.opponent == Player.GOTE
    assert Player.GOTE.opponent == Player.SENTE


def test_14_piece_types() -> None:
    assert len(PieceType) == 14


def test_promotion_map() -> None:
    assert PROMOTION_MAP[PieceType.PAWN] == PieceType.PRO_PAWN
    assert PROMOTION_MAP[PieceType.BISHOP] == PieceType.HORSE
    assert PROMOTION_MAP[PieceType.ROOK] == PieceType.DRAGON
    assert PieceType.GOLD not in PROMOTION_MAP
    assert PieceType.KING not in PROMOTION_MAP


def test_unpromotion_map() -> None:
    assert UNPROMOTION_MAP[PieceType.PRO_PAWN] == PieceType.PAWN
    assert UNPROMOTION_MAP[PieceType.HORSE] == PieceType.BISHOP
    assert UNPROMOTION_MAP[PieceType.DRAGON] == PieceType.ROOK


def test_hand_piece_types() -> None:
    assert len(HAND_PIECE_TYPES) == 7
    assert PieceType.KING not in HAND_PIECE_TYPES
    assert PieceType.PAWN in HAND_PIECE_TYPES
    assert PieceType.ROOK in HAND_PIECE_TYPES
