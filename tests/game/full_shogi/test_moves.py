"""Tests for full shogi move generation."""

from __future__ import annotations

from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.moves import (
    ACTION_SPACE,
    _is_in_check,
    apply_move,
    decode_move,
    encode_board_move,
    encode_drop_move,
    legal_moves,
)
from shogi_ai.game.full_shogi.types import (
    COLS,
    NUM_SQUARES,
    PieceType,
    Player,
)


class TestMoveEncoding:
    def test_board_move_roundtrip(self) -> None:
        move = encode_board_move(0, 9)  # (0,0) → (1,0)
        decoded = decode_move(move)
        assert decoded["type"] == "board"
        assert decoded["from"] == (0, 0)
        assert decoded["to"] == (1, 0)
        assert decoded["promote"] is False

    def test_promotion_move_roundtrip(self) -> None:
        move = encode_board_move(18, 9, promote=True)  # (2,0) → (1,0)
        decoded = decode_move(move)
        assert decoded["type"] == "board"
        assert decoded["promote"] is True

    def test_drop_move_roundtrip(self) -> None:
        move = encode_drop_move(PieceType.PAWN, 40)  # Drop pawn to center
        decoded = decode_move(move)
        assert decoded["type"] == "drop"
        assert decoded["piece_type"] == PieceType.PAWN

    def test_action_space_size(self) -> None:
        assert ACTION_SPACE == 13689


class TestInitialLegalMoves:
    def test_initial_move_count(self) -> None:
        """Initial position should have 30 legal moves for Sente."""
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        assert len(moves) == 30

    def test_all_moves_are_valid_indices(self) -> None:
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            assert 0 <= move < ACTION_SPACE


class TestNifuRestriction:
    def test_cannot_drop_pawn_in_column_with_pawn(self) -> None:
        """二歩: Cannot drop pawn in column that already has own pawn."""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        # Kings
        squares[0 * COLS + 4] = Piece(PieceType.KING, Player.GOTE)
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        # Sente pawn in column 0
        squares[6 * COLS + 0] = Piece(PieceType.PAWN, Player.SENTE)
        board = Board(
            squares=tuple(squares),
            hands=((PieceType.PAWN,), ()),
        )

        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            decoded = decode_move(move)
            if decoded["type"] == "drop" and decoded["piece_type"] == PieceType.PAWN:
                _, drop_col = decoded["to"]
                assert drop_col != 0, "Should not be able to drop pawn in col 0"


class TestDeadPieceRestriction:
    def test_cannot_drop_pawn_on_last_rank(self) -> None:
        """行き所のない駒: Cannot drop pawn on opponent's back rank."""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0 * COLS + 4] = Piece(PieceType.KING, Player.GOTE)
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        board = Board(
            squares=tuple(squares),
            hands=((PieceType.PAWN,), ()),
        )

        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            decoded = decode_move(move)
            if decoded["type"] == "drop" and decoded["piece_type"] == PieceType.PAWN:
                drop_row, _ = decoded["to"]
                assert drop_row != 0, "Should not drop pawn on row 0"

    def test_cannot_drop_knight_on_last_two_ranks(self) -> None:
        """行き所のない駒: Cannot drop knight on last 2 ranks."""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0 * COLS + 4] = Piece(PieceType.KING, Player.GOTE)
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        board = Board(
            squares=tuple(squares),
            hands=((PieceType.KNIGHT,), ()),
        )

        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            decoded = decode_move(move)
            if decoded["type"] == "drop" and decoded["piece_type"] == PieceType.KNIGHT:
                drop_row, _ = decoded["to"]
                assert drop_row > 1, f"Should not drop knight on row {drop_row}"


class TestCheckRestriction:
    def test_cannot_leave_king_in_check(self) -> None:
        """王手放置禁止: No legal move should leave own king in check."""
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        # All initial moves are legal (no check issues from initial position)
        assert len(moves) > 0
        # Verify by applying each move — king should not be capturable
        for move in moves:
            new_board = apply_move(board, Player.SENTE, move)
            assert not _is_in_check(new_board, Player.SENTE)


class TestPromotion:
    def test_pawn_must_promote_on_last_rank(self) -> None:
        """Pawn must promote when reaching opponent's back rank."""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0 * COLS + 4] = Piece(PieceType.KING, Player.GOTE)
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        # Sente pawn on row 1, can move to row 0 (must promote)
        squares[1 * COLS + 0] = Piece(PieceType.PAWN, Player.SENTE)
        board = Board(squares=tuple(squares), hands=((), ()))

        moves = legal_moves(board, Player.SENTE)
        pawn_moves = []
        for move in moves:
            decoded = decode_move(move)
            if decoded["type"] == "board" and decoded["from"] == (1, 0):
                pawn_moves.append(decoded)

        # There should be exactly one move and it must be promotion
        assert len(pawn_moves) == 1
        assert pawn_moves[0]["promote"] is True

    def test_pawn_can_optionally_promote_in_zone(self) -> None:
        """Pawn in promotion zone (not last rank) can choose to promote."""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0 * COLS + 4] = Piece(PieceType.KING, Player.GOTE)
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        # Sente pawn on row 3, can move to row 2 (promotion zone)
        squares[3 * COLS + 0] = Piece(PieceType.PAWN, Player.SENTE)
        board = Board(squares=tuple(squares), hands=((), ()))

        moves = legal_moves(board, Player.SENTE)
        pawn_moves = []
        for move in moves:
            decoded = decode_move(move)
            if decoded["type"] == "board" and decoded["from"] == (3, 0):
                pawn_moves.append(decoded)

        # Should have both promote and non-promote options
        promotes = [m for m in pawn_moves if m["promote"]]
        non_promotes = [m for m in pawn_moves if not m["promote"]]
        assert len(promotes) == 1
        assert len(non_promotes) == 1
