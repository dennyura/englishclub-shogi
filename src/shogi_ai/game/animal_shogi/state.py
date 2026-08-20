"""GameState implementation for どうぶつしょうぎ.

どうぶつしょうぎの対局状態（ゲームツリーのノード）。
Board クラスが盤面データを持ち、AnimalShogiState が対局ルール・勝敗判定を担当する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shogi_ai.game.animal_shogi.board import Board
from shogi_ai.game.animal_shogi.moves import ACTION_SPACE, DROP_OFFSET
from shogi_ai.game.animal_shogi.moves import apply_move as _apply_move
from shogi_ai.game.animal_shogi.moves import legal_moves as _legal_moves
from shogi_ai.game.animal_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    ROWS,
    Player,
)

# テンソルチャンネルインデックス（to_tensor_planes で使用）
_OPP_PIECE_CH = 5  # 相手駒チャンネルの開始インデックス（ch.5〜9）
_HAND_CH = 10  # 持ち駒チャンネルの開始インデックス（ch.10〜12）
_TURN_CH = 13  # 手番インジケータチャンネル


@dataclass(frozen=True)  # イミュータブル: apply_move() は新しいオブジェクトを返す
class AnimalShogiState:
    """Immutable game state for どうぶつしょうぎ.

    どうぶつしょうぎの対局状態。GameState プロトコルを実装する。

    Terminal conditions（終局条件）:
    1. ライオン取り: 相手のライオンを取った → 取ったプレイヤーの勝ち
    2. トライルール: ライオンを相手の後ろ段に移動し、取られない → そのプレイヤーの勝ち
    3. 合法手なし: 手が存在しない → そのプレイヤーの負け（スタレメート）
    """

    board: Board = field(default_factory=Board)
    _current_player: Player = Player.SENTE  # 先手から開始
    _move_count: int = 0  # 手数（記録用）

    @property
    def action_space_size(self) -> int:
        """行動空間のサイズ（どうぶつしょうぎは 180 手）。"""
        return ACTION_SPACE

    @property
    def current_player(self) -> int:
        """現在の手番プレイヤー（0=先手, 1=後手）。"""
        return self._current_player.value

    @property
    def is_terminal(self) -> bool:
        """ゲームが終局ならば True。"""
        return self.winner is not None or len(self.legal_moves()) == 0

    @property
    def winner(self) -> int | None:
        """勝者を返す。対局中または引き分けは None。

        判定順序:
        1. ライオン取り → ライオンを失ったプレイヤーの負け
        2. トライルール → 前手番プレイヤーが条件を満たせば勝ち
        3. 合法手なし → 現在プレイヤーの負け
        """
        # 1. ライオン取り判定
        for player in Player:
            if self.board.find_lion(player) is None:
                return player.opponent.value  # ライオンを失った → 相手の勝ち

        # 2. トライルール判定
        # 前手番プレイヤーがライオンを相手陣地後段に置いていて、取られなければ勝ち
        prev_player = self._current_player.opponent
        lion_idx = self.board.find_lion(prev_player)
        if lion_idx is not None:
            lion_row = lion_idx // COLS
            target_row = 0 if prev_player == Player.SENTE else ROWS - 1
            if lion_row == target_row:
                # 現プレイヤーがそのライオンを取れない → トライ成功
                if not self._can_capture_lion(self._current_player, lion_idx):
                    return prev_player.value

        # 3. 合法手なし → 現プレイヤーの負け
        if len(self.legal_moves()) == 0:
            return self._current_player.opponent.value

        return None  # まだ対局中

    def legal_moves(self) -> list[int]:
        """合法手のリストを返す。"""
        return _legal_moves(self.board, self._current_player)

    def apply_move(self, move: int) -> AnimalShogiState:
        """手を適用して新しい対局状態を返す。

        元の状態は変更せず、新しい AnimalShogiState を生成する。
        手番は自動的に切り替わる。
        """
        new_board = _apply_move(self.board, self._current_player, move)
        return AnimalShogiState(
            board=new_board,
            _current_player=self._current_player.opponent,  # 手番交代
            _move_count=self._move_count + 1,
        )

    def to_tensor_planes(self) -> torch.Tensor:
        """Convert to tensor planes for neural network input.

        局面をニューラルネットワーク入力用テンソルに変換する。

        Planes（チャンネル）の構成（合計14チャンネル）:
        ch.0-4:   現プレイヤーの駒（Chick, Giraffe, Elephant, Lion, Hen）
        ch.5-9:   相手プレイヤーの駒
        ch.10-12: 現プレイヤーの持ち駒数（Chick, Giraffe, Elephant）
        ch.13:    手番インジケータ（先手番なら全1、後手番なら全0）

        AlphaZero は常に「現プレイヤーの視点」でテンソルを作る。
        これにより先手・後手を区別せず同じニューラルネットを使える。
        """
        planes = torch.zeros(14, ROWS, COLS)
        cp = self._current_player

        # 盤上の駒をテンソルに配置
        for idx, piece in enumerate(self.board.squares):
            if piece is None:
                continue
            r, c = idx // COLS, idx % COLS
            if piece.owner == cp:
                planes[piece.piece_type.value, r, c] = 1.0  # 自分の駒
            else:
                planes[_OPP_PIECE_CH + piece.piece_type.value, r, c] = 1.0  # 相手の駒

        # 現プレイヤーの持ち駒数をチャンネルに記録
        for i, pt in enumerate(HAND_PIECE_TYPES):
            count = self.board.hands[cp.value].count(pt)
            if count > 0:
                planes[_HAND_CH + i, :, :] = float(count)  # 全マスに枚数を設定

        # 手番インジケータ（先手番なら全マス1.0）
        if cp == Player.SENTE:
            planes[_TURN_CH, :, :] = 1.0

        return planes

    def _can_capture_lion(self, player: Player, lion_idx: int) -> bool:
        """Check if player can capture the piece at lion_idx.

        プレイヤーが lion_idx にあるライオンを取れるか判定する。
        トライルール判定で「取られるかどうか」を調べるために使用。
        """
        moves = _legal_moves(self.board, player)
        for move in moves:
            if move < DROP_OFFSET:  # 盤上の手のみ（持ち駒打ちはライオンを取れない）
                to_idx = move % 12
                if to_idx == lion_idx:
                    return True
        return False
