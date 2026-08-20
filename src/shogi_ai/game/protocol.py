"""GameState protocol — all games implement this interface.

ゲーム状態の共通インタフェース（プロトコル）。

どうぶつしょうぎ・本将棋・その他のゲームがこのプロトコルを実装することで、
MCTS やミニマックスなどのエンジンがゲームに依存せず動作できる。
これを「ポリモーフィズム」または「ダックタイピング」と呼ぶ。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable  # isinstance() でのランタイムチェックを有効にする
class GameState(Protocol):
    """Common interface for all board game states.

    すべてのボードゲームが実装すべき共通インタフェース。
    このプロトコルを実装したクラスなら何でも MCTS やミニマックスで使える。

    重要: apply_move() は新しい状態を返す（イミュータブル設計）。
    イミュータブルにすることで、探索木のノードを安全に共有できる。
    """

    @property
    def current_player(self) -> int:
        """現在手番のプレイヤー（0=先手, 1=後手）を返す。"""
        ...

    @property
    def is_terminal(self) -> bool:
        """ゲームが終了していれば True を返す。"""
        ...

    @property
    def winner(self) -> int | None:
        """勝者（0 or 1）を返す。引き分けや対局中は None。"""
        ...

    def legal_moves(self) -> list[int]:
        """合法手のインデックスリストを返す。"""
        ...

    def apply_move(self, move: int) -> GameState:
        """手を適用した新しい状態を返す（元の状態は変化しない）。"""
        ...

    @property
    def action_space_size(self) -> int:
        """行動空間のサイズ（可能な手の総数）を返す。"""
        ...

    def to_tensor_planes(self) -> torch.Tensor:
        """局面をニューラルネットワーク入力用テンソルに変換する。"""
        ...
