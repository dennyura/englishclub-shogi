"""Tests for minimax with FullShogiState (本将棋).

本将棋エンジンのミニマックス動作テスト。
どうぶつしょうぎと同じ GameState プロトコルで動くことを確認する。
"""

from __future__ import annotations

from shogi_ai.engine.minimax import (
    _FULL_SHOGI_BASE_VALUE,
    _FULL_SHOGI_HAND_BONUS,
    _FULL_SHOGI_PROMOTE_BONUS,
    evaluate,
    minimax_move,
)
from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.moves import ACTION_SPACE
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.game.full_shogi.types import HAND_PIECE_TYPES, NUM_SQUARES, PieceType, Player


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

    def test_all_base_piece_types_have_configured_values(self) -> None:
        """本将棋の基礎価値と7種のボーナスが各テーブルに登録されている。"""
        expected = set(HAND_PIECE_TYPES)
        assert set(_FULL_SHOGI_BASE_VALUE) == expected | {PieceType.KING}
        assert set(_FULL_SHOGI_PROMOTE_BONUS) == expected
        assert set(_FULL_SHOGI_HAND_BONUS) == expected

    def test_full_shogi_king_value_is_10000(self) -> None:
        """玉の基礎価値は10000。"""
        assert _FULL_SHOGI_BASE_VALUE[PieceType.KING] == 10000.0

    def test_full_shogi_base_value_is_used_for_board_piece(self) -> None:
        """未成駒はBASE_VALUEで評価される。"""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0] = Piece(PieceType.KING, Player.GOTE)
        squares[-1] = Piece(PieceType.KING, Player.SENTE)
        squares[40] = Piece(PieceType.BISHOP, Player.SENTE)
        state = FullShogiState(board=Board(squares=tuple(squares)))

        assert evaluate(state) == 1040.0

    def test_full_shogi_promoted_value_includes_promotion_bonus(self) -> None:
        """成り駒は未成駒の価値と成りボーナスの合計で評価される。"""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0] = Piece(PieceType.KING, Player.GOTE)
        squares[-1] = Piece(PieceType.KING, Player.SENTE)
        squares[40] = Piece(PieceType.PRO_PAWN, Player.SENTE)
        state = FullShogiState(board=Board(squares=tuple(squares)))

        assert evaluate(state) == 520.0

    def test_full_shogi_hand_value_is_used_for_hand_piece(self) -> None:
        """持ち駒はBASE_VALUEとHAND_BONUSの合計で評価される。"""
        squares: list[Piece | None] = [None] * NUM_SQUARES
        squares[0] = Piece(PieceType.KING, Player.GOTE)
        squares[-1] = Piece(PieceType.KING, Player.SENTE)
        board = Board(squares=tuple(squares), hands=((PieceType.ROOK,), ()))
        state = FullShogiState(board=board)

        assert evaluate(state) == 1270.0
