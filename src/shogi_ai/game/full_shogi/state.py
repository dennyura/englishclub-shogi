"""GameState implementation for 本将棋 (Full Shogi).

本将棋の対局状態（ゲームツリーのノード）。
どうぶつしょうぎと同じ設計パターンを9×9盤に適用している。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shogi_ai.game.full_shogi.board import Board
from shogi_ai.game.full_shogi.moves import ACTION_SPACE
from shogi_ai.game.full_shogi.moves import apply_move as _apply_move
from shogi_ai.game.full_shogi.moves import legal_moves as _legal_moves
from shogi_ai.game.full_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    ROWS,
    Player,
)


@dataclass(frozen=True)
class FullShogiState:
    """Immutable game state for 本将棋 (9x9).

    本将棋の対局状態。GameState プロトコルを実装する。

    Terminal conditions（終局条件）:
    1. 王将取り: 本来は合法手フィルタで王手放置を防ぐため起きない
    2. 合法手なし（詰み・千日手）: 現プレイヤーの負け
    """

    board: Board = field(default_factory=Board)
    _current_player: Player = Player.SENTE
    _move_count: int = 0

    @property
    def action_space_size(self) -> int:
        """行動空間のサイズ（本将棋は 13689 手）。"""
        return ACTION_SPACE

    @property
    def current_player(self) -> int:
        """現在の手番プレイヤー（0=先手, 1=後手）。"""
        return self._current_player.value

    @property
    def is_terminal(self) -> bool:
        """ゲームが終局ならば True。

        王将がいない（取られた）または合法手がない場合に終局。
        """
        # 王将がいない場合（正常な合法手フィルタなら起きない）
        for player in Player:
            if self.board.find_king(player) is None:
                return True
        return len(self.legal_moves()) == 0

    @property
    def winner(self) -> int | None:
        """勝者を返す。対局中は None。"""
        # 王将がいない場合（異常終了）
        for player in Player:
            if self.board.find_king(player) is None:
                return player.opponent.value

        # 合法手なし = 詰み → 現プレイヤーの負け
        if len(self.legal_moves()) == 0:
            return self._current_player.opponent.value

        return None

    def legal_moves(self) -> list[int]:
        """合法手のリストを返す。"""
        return _legal_moves(self.board, self._current_player)

    def apply_move(self, move: int) -> FullShogiState:
        """手を適用して新しい対局状態を返す。"""
        new_board = _apply_move(self.board, self._current_player, move)
        return FullShogiState(
            board=new_board,
            _current_player=self._current_player.opponent,  # 手番交代
            _move_count=self._move_count + 1,
        )

    def to_tensor_planes(self) -> torch.Tensor:
        """Convert to tensor planes for neural network input.

        局面をニューラルネットワーク入力用テンソルに変換する（43チャンネル）。

        Planes（チャンネル）の構成:
        ch.0-13:  現プレイヤーの駒（14駒種）
        ch.14-27: 相手プレイヤーの駒（14駒種）
        ch.28-34: 現プレイヤーの持ち駒数（7種）
        ch.35-41: 相手プレイヤーの持ち駒数（7種）
        ch.42:    手番インジケータ（先手番なら全1）

        合計: 14+14+7+7+1 = 43 チャンネル
        """
        planes = torch.zeros(43, ROWS, COLS)
        cp = self._current_player

        # 盤上の駒をテンソルに配置
        for idx, piece in enumerate(self.board.squares):
            if piece is None:
                continue
            r, c = idx // COLS, idx % COLS
            if piece.owner == cp:
                planes[piece.piece_type.value, r, c] = 1.0  # 自分の駒
            else:
                planes[14 + piece.piece_type.value, r, c] = 1.0  # 相手の駒

        # 持ち駒数をチャンネルに記録（自分・相手それぞれ7種）
        for i, pt in enumerate(HAND_PIECE_TYPES):
            cp_count = self.board.hands[cp.value].count(pt)
            opp_count = self.board.hands[cp.opponent.value].count(pt)
            if cp_count > 0:
                planes[28 + i, :, :] = float(cp_count)
            if opp_count > 0:
                planes[35 + i, :, :] = float(opp_count)

        # 手番インジケータ
        if cp == Player.SENTE:
            planes[42, :, :] = 1.0

        return planes
