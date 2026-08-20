"""Tests for minimax with FullShogiState (本将棋).

本将棋エンジンのミニマックス動作テスト。
どうぶつしょうぎと同じ GameState プロトコルで動くことを確認する。
"""

from __future__ import annotations

from shogi_ai.engine.minimax import evaluate, minimax_move
from shogi_ai.game.full_shogi.moves import ACTION_SPACE
from shogi_ai.game.full_shogi.state import FullShogiState


class TestMinimaxWithFullShogi:
    def test_initial_position_evaluate_near_zero(self) -> None:
        """初期局面の評価値は先後対称なのでほぼ0になる。"""
        state = FullShogiState()
        score = evaluate(state)
        assert -10.0 <= score <= 10.0

    def test_minimax_returns_legal_move_depth1(self) -> None:
        """深さ1のミニマックスが合法手を返す。"""
        state = FullShogiState()
        move = minimax_move(state, depth=1)
        assert move in state.legal_moves()
        assert 0 <= move < ACTION_SPACE

    def test_minimax_returns_legal_move_depth2(self) -> None:
        """深さ2のミニマックスが合法手を返す（本将棋は合法手が多いため低深度）。"""
        state = FullShogiState()
        move = minimax_move(state, depth=2)
        assert move in state.legal_moves()
        assert 0 <= move < ACTION_SPACE

    def test_apply_minimax_move_advances_game(self) -> None:
        """ミニマックスの手を適用すると手番が変わる。"""
        state = FullShogiState()
        assert state.current_player == 0  # 先手
        move = minimax_move(state, depth=1)
        next_state = state.apply_move(move)
        assert next_state.current_player == 1  # 後手に変わる

    def test_protocol_compatibility(self) -> None:
        """FullShogiState が GameState プロトコルに準拠している。"""
        from shogi_ai.game.protocol import GameState

        state = FullShogiState()
        assert isinstance(state, GameState)
        # プロトコルの全プロパティ・メソッドが動作する
        assert isinstance(state.current_player, int)
        assert isinstance(state.is_terminal, bool)
        assert isinstance(state.legal_moves(), list)
        assert isinstance(state.action_space_size, int)
        assert state.action_space_size == ACTION_SPACE
