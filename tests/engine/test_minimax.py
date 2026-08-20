"""Tests for minimax search engine."""

from __future__ import annotations

import pytest

from shogi_ai.engine.minimax import evaluate, minimax_move, negamax
from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.moves import ACTION_SPACE
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import COLS, PieceType, Player


def _make_state(
    squares: list[Piece | None],
    current_player: Player = Player.SENTE,
) -> AnimalShogiState:
    """Helper: create a state from a flat list of squares."""
    board = Board(squares=tuple(squares), hands=((), ()))
    return AnimalShogiState(board=board, _current_player=current_player)


class TestEvaluate:
    def test_initial_position_roughly_zero(self) -> None:
        state = AnimalShogiState()
        score = evaluate(state)
        # Symmetric position → close to 0
        assert -1.0 <= score <= 1.0

    def test_missing_opponent_lion_very_high(self) -> None:
        # Sente captured Gote's lion → terminal, Sente wins
        squares: list[Piece | None] = [None] * 12
        squares[10] = Piece(PieceType.LION, Player.SENTE)  # Sente lion at (3,1)
        state = _make_state(squares, Player.SENTE)
        score = evaluate(state)
        assert score > 50  # Very high for current player (Sente)


class TestNegamax:
    def test_returns_valid_move(self) -> None:
        state = AnimalShogiState()
        move, score = negamax(state, depth=2, alpha=float("-inf"), beta=float("inf"))
        assert 0 <= move < ACTION_SPACE
        assert move in state.legal_moves()

    def test_finds_checkmate_in_one(self) -> None:
        """Set up a position where Sente can capture Gote's lion in one move."""
        squares: list[Piece | None] = [None] * 12
        # Gote lion at (0,1), Sente lion at (3,1)
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        # Sente giraffe at (1,1) can capture Gote's lion at (0,1)
        squares[1 * COLS + 1] = Piece(PieceType.GIRAFFE, Player.SENTE)
        state = _make_state(squares, Player.SENTE)

        move, score = negamax(state, depth=2, alpha=float("-inf"), beta=float("inf"))
        # Score should be very high (winning)
        assert score > 50
        # The move should capture the lion
        from shogi_ai.game.animal_shogi.moves import decode_move

        decoded = decode_move(move)
        assert decoded["to"] == (0, 1)  # Gote's lion position

    def test_depth_zero_returns_eval(self) -> None:
        state = AnimalShogiState()
        move, score = negamax(state, depth=0, alpha=float("-inf"), beta=float("inf"))
        # At depth 0, move is -1 (no search)
        assert move == -1

    def test_terminal_state_returns_correct_score(self) -> None:
        """A terminal state should return extreme score."""
        squares: list[Piece | None] = [None] * 12
        # Only Sente lion → Gote lost
        squares[10] = Piece(PieceType.LION, Player.SENTE)
        state = _make_state(squares, Player.SENTE)
        assert state.is_terminal
        move, score = negamax(state, depth=3, alpha=float("-inf"), beta=float("inf"))
        assert move == -1
        assert score > 100


class TestMinimaxMove:
    def test_returns_legal_move(self) -> None:
        state = AnimalShogiState()
        move = minimax_move(state, depth=3)
        assert move in state.legal_moves()

    def test_captures_lion_if_possible(self) -> None:
        """When lion capture is available, minimax should find it."""
        squares: list[Piece | None] = [None] * 12
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        squares[1 * COLS + 1] = Piece(PieceType.GIRAFFE, Player.SENTE)
        state = _make_state(squares, Player.SENTE)

        move = minimax_move(state, depth=1)
        from shogi_ai.game.animal_shogi.moves import decode_move

        decoded = decode_move(move)
        assert decoded["to"] == (0, 1)


class TestMinimaxVsRandom:
    @pytest.mark.slow
    def test_minimax_wins_vs_random(self) -> None:
        """Minimax (depth=4) should win >90% vs random player."""
        wins = 0
        num_games = 100

        for _ in range(num_games):
            state = AnimalShogiState()
            move_count = 0
            while not state.is_terminal and move_count < 200:
                if state.current_player == 0:  # Sente = minimax
                    move = minimax_move(state, depth=4)
                else:  # Gote = random
                    move = random_move(state)
                state = state.apply_move(move)
                move_count += 1

            if state.winner == 0:  # Sente (minimax) wins
                wins += 1

        win_rate = wins / num_games
        assert win_rate > 0.90, f"Minimax win rate {win_rate:.0%} < 90%"


class TestAlphaBetaPruning:
    def test_deeper_search_does_not_crash(self) -> None:
        """Depth 6 should complete in reasonable time with alpha-beta."""
        state = AnimalShogiState()
        move, score = negamax(state, depth=6, alpha=float("-inf"), beta=float("inf"))
        assert 0 <= move < ACTION_SPACE
