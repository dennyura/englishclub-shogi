"""Tests for full shogi state."""

from __future__ import annotations

from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.moves import ACTION_SPACE
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.game.full_shogi.types import COLS, NUM_SQUARES, PieceType, Player
from shogi_ai.game.protocol import GameState


class TestProtocolCompliance:
    def test_implements_game_state(self) -> None:
        state = FullShogiState()
        assert isinstance(state, GameState)

    def test_action_space_size(self) -> None:
        state = FullShogiState()
        assert state.action_space_size == ACTION_SPACE


class TestInitialState:
    def test_sente_starts(self) -> None:
        state = FullShogiState()
        assert state.current_player == 0

    def test_not_terminal(self) -> None:
        state = FullShogiState()
        assert not state.is_terminal

    def test_winner_none(self) -> None:
        state = FullShogiState()
        assert state.winner is None

    def test_has_30_legal_moves(self) -> None:
        state = FullShogiState()
        assert len(state.legal_moves()) == 30


class TestApplyMove:
    def test_player_alternates(self) -> None:
        state = FullShogiState()
        moves = state.legal_moves()
        new_state = state.apply_move(moves[0])
        assert new_state.current_player == 1

    def test_immutability(self) -> None:
        state = FullShogiState()
        moves = state.legal_moves()
        new_state = state.apply_move(moves[0])
        assert state.current_player == 0  # Original unchanged
        assert new_state.current_player == 1


class TestTerminal:
    def test_king_missing_is_terminal(self) -> None:
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[8 * COLS + 4] = Piece(PieceType.KING, Player.SENTE)
        # No gote king
        board = Board(squares=tuple(squares), hands=((), ()))
        state = FullShogiState(board=board, _current_player=Player.SENTE)
        assert state.is_terminal
        assert state.winner == 0  # Sente wins


class TestTensorPlanes:
    def test_shape(self) -> None:
        state = FullShogiState()
        planes = state.to_tensor_planes()
        assert planes.shape == (43, 9, 9)

    def test_sente_king_plane(self) -> None:
        state = FullShogiState()
        planes = state.to_tensor_planes()
        # King is PieceType 7, at (8, 4)
        assert planes[7, 8, 4] == 1.0

    def test_turn_indicator(self) -> None:
        state = FullShogiState()
        planes = state.to_tensor_planes()
        # Sente → plane 42 all ones
        assert planes[42].sum() == 81.0


class TestMultipleMoves:
    def test_several_moves_no_crash(self) -> None:
        """Play several moves without crashing."""
        state = FullShogiState()
        for _ in range(10):
            moves = state.legal_moves()
            if not moves:
                break
            state = state.apply_move(moves[0])
        # Should not crash
        assert state.current_player in (0, 1)
