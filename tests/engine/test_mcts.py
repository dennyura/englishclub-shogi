"""Tests for MCTS search."""

from __future__ import annotations

import torch

from shogi_ai.engine.mcts import MCTS, MCTSConfig, MCTSNode
from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.moves import ACTION_SPACE
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import COLS, PieceType, Player
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork


def _make_network() -> DualHeadNetwork:
    net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
    net.eval()
    return net


class TestMCTSNode:
    def test_initial_q_value(self) -> None:
        node = MCTSNode()
        assert node.q_value == 0.0

    def test_q_value_after_visits(self) -> None:
        node = MCTSNode(visit_count=4, total_value=2.0)
        assert node.q_value == 0.5


class TestMCTSSearch:
    def test_returns_valid_probabilities(self) -> None:
        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=10))
        state = AnimalShogiState()
        probs = mcts.search(state)

        assert len(probs) == ACTION_SPACE
        assert abs(sum(probs) - 1.0) < 0.01
        # Only legal moves should have non-zero probability
        legal = set(state.legal_moves())
        for i, p in enumerate(probs):
            if i not in legal:
                assert p == 0.0 or abs(p) < 1e-6

    def test_finds_checkmate_in_one(self) -> None:
        """MCTS should strongly prefer capturing the lion."""
        squares: list[Piece | None] = [None] * 12
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        squares[1 * COLS + 1] = Piece(PieceType.GIRAFFE, Player.SENTE)
        board = Board(squares=tuple(squares), hands=((), ()))
        state = AnimalShogiState(board=board, _current_player=Player.SENTE)

        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=50))
        probs = mcts.search(state)

        # The winning move (giraffe captures lion)
        from shogi_ai.game.animal_shogi.moves import encode_board_move

        winning_move = encode_board_move(1 * COLS + 1, 0 * COLS + 1)
        assert probs[winning_move] > 0.5

    def test_terminal_state_returns_zeros(self) -> None:
        """Terminal state with no legal moves returns all zeros."""
        squares: list[Piece | None] = [None] * 12
        squares[10] = Piece(PieceType.LION, Player.SENTE)
        board = Board(squares=tuple(squares), hands=((), ()))
        state = AnimalShogiState(board=board, _current_player=Player.SENTE)
        assert state.is_terminal

        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=10))
        probs = mcts.search(state)
        assert sum(probs) == 0.0

    def test_deterministic_temperature(self) -> None:
        """Temperature=0 should give deterministic selection."""
        net = _make_network()
        torch.manual_seed(42)
        mcts = MCTS(net, MCTSConfig(num_simulations=20, temperature=0))
        state = AnimalShogiState()
        probs = mcts.search(state)
        # Exactly one move should have probability 1.0
        assert sum(1 for p in probs if p > 0.99) == 1
