"""Random player — selects a legal move uniformly at random.

ランダムプレイヤー: 合法手の中からランダムに手を選ぶ。

用途:
- 実装の動作確認（ルールが正しく実装されているかテスト）
- ベースラインとの対戦（ランダムに勝てないAIは弱すぎる）
- 自己対局の初期段階（ネットワーク未学習時の代替手法）
"""

from __future__ import annotations

import random

from shogi_ai.game.protocol import GameState


def random_move(state: GameState) -> int:
    """Return a random legal move.

    合法手の中から一様ランダムで1手を返す。
    合法手がない場合は ValueError を送出する（終局局面では呼ばれないはず）。
    """
    moves = state.legal_moves()
    if not moves:
        raise ValueError("No legal moves available")
    return random.choice(moves)  # 一様ランダムサンプリング
