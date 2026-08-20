"""Types and constants for どうぶつしょうぎ (Animal Shogi).

どうぶつしょうぎの基本型・定数定義。
盤面は 3列 × 4行（12マス）で、駒は5種類。
"""

from __future__ import annotations

from enum import IntEnum, unique

# 盤面のサイズ: 3列 × 4行
COLS = 3
ROWS = 4


@unique
class Player(IntEnum):
    """Player identifiers.

    先手（SENTE）は下側から上に向かって進む。
    後手（GOTE）は上側から下に向かって進む。
    """

    SENTE = 0  # 先手 (first player, moves upward)
    GOTE = 1  # 後手 (second player, moves downward)

    @property
    def opponent(self) -> Player:
        """相手プレイヤーを返す。0↔1 の切り替え。"""
        return Player(1 - self.value)


@unique
class PieceType(IntEnum):
    """Piece types in どうぶつしょうぎ.

    どうぶつしょうぎの駒種（5種類）。
    値は to_tensor_planes() でのチャンネルインデックスに対応する。
    """

    CHICK = 0  # ひよこ — 1マス前にしか動けない
    GIRAFFE = 1  # きりん — 縦横1マス（飛車的、ただし1マス）
    ELEPHANT = 2  # ぞう   — 斜め1マス（角的、ただし1マス）
    LION = 3  # ライオン — 全方向1マス（王将と同じ動き）
    HEN = 4  # にわとり — 成りひよこ（金将的な動き）


# 移動方向の定義: (行の変化, 列の変化) のリスト
# 先手（SENTE）視点で定義。後手（GOTE）は行方向を反転して使う。
PIECE_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.CHICK: [(-1, 0)],  # 1マス前のみ
    PieceType.GIRAFFE: [(-1, 0), (1, 0), (0, -1), (0, 1)],  # 縦横4方向
    PieceType.ELEPHANT: [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # 斜め4方向
    PieceType.LION: [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],  # 全8方向
    PieceType.HEN: [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, 0),
    ],  # 金将と同じ動き（斜め後ろを除く6方向）
}

# 持ち駒として使える駒種（ライオン・にわとりは持ち駒にならない）
# 取られたにわとりはひよこに戻って相手の持ち駒になる
HAND_PIECE_TYPES = [PieceType.CHICK, PieceType.GIRAFFE, PieceType.ELEPHANT]
