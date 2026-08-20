"""Tests for full shogi board."""

from __future__ import annotations

from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.types import (
    COLS,
    NUM_SQUARES,
    PieceType,
    Player,
)


class TestInitialPosition:
    def test_81_squares(self) -> None:
        board = Board()
        assert len(board.squares) == NUM_SQUARES

    def test_sente_king_at_row8_col4(self) -> None:
        board = Board()
        piece = board.piece_at(8, 4)
        assert piece is not None
        assert piece.piece_type == PieceType.KING
        assert piece.owner == Player.SENTE

    def test_gote_king_at_row0_col4(self) -> None:
        board = Board()
        piece = board.piece_at(0, 4)
        assert piece is not None
        assert piece.piece_type == PieceType.KING
        assert piece.owner == Player.GOTE

    def test_sente_rook_at_row7_col7(self) -> None:
        board = Board()
        piece = board.piece_at(7, 7)
        assert piece is not None
        assert piece.piece_type == PieceType.ROOK
        assert piece.owner == Player.SENTE

    def test_gote_bishop_at_row1_col7(self) -> None:
        board = Board()
        piece = board.piece_at(1, 7)
        assert piece is not None
        assert piece.piece_type == PieceType.BISHOP
        assert piece.owner == Player.GOTE

    def test_sente_pawns_on_row6(self) -> None:
        board = Board()
        for c in range(COLS):
            piece = board.piece_at(6, c)
            assert piece is not None
            assert piece.piece_type == PieceType.PAWN
            assert piece.owner == Player.SENTE

    def test_gote_pawns_on_row2(self) -> None:
        board = Board()
        for c in range(COLS):
            piece = board.piece_at(2, c)
            assert piece is not None
            assert piece.piece_type == PieceType.PAWN
            assert piece.owner == Player.GOTE

    def test_empty_squares_in_middle(self) -> None:
        board = Board()
        for r in range(3, 6):
            for c in range(COLS):
                assert board.piece_at(r, c) is None

    def test_empty_hands(self) -> None:
        board = Board()
        assert board.hands == ((), ())

    def test_piece_count(self) -> None:
        board = Board()
        pieces = [p for p in board.squares if p is not None]
        assert len(pieces) == 40  # 20 per side


class TestBoardOperations:
    def test_set_piece(self) -> None:
        board = Board()
        new_board = board.set_piece(4, 4, Piece(PieceType.GOLD, Player.SENTE))
        assert new_board.piece_at(4, 4) is not None
        assert board.piece_at(4, 4) is None  # Original unchanged

    def test_add_to_hand_reverts_promotion(self) -> None:
        board = Board()
        new_board = board.add_to_hand(Player.SENTE, PieceType.PRO_PAWN)
        assert PieceType.PAWN in new_board.hands[0]
        assert PieceType.PRO_PAWN not in new_board.hands[0]

    def test_find_king(self) -> None:
        board = Board()
        sente_king = board.find_king(Player.SENTE)
        gote_king = board.find_king(Player.GOTE)
        assert sente_king == 8 * COLS + 4
        assert gote_king == 0 * COLS + 4

    def test_count_pawns_in_column(self) -> None:
        board = Board()
        # Column 4 has one sente pawn (row 6) and one gote pawn (row 2)
        assert board.count_pawns_in_column(Player.SENTE, 4) == 1
        assert board.count_pawns_in_column(Player.GOTE, 4) == 1
