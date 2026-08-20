"""Types and constants for 本将棋 (Full Shogi, 9x9).

本将棋（9×9盤）の基本型・定数定義。
駒は14種類（未成7種 + 成り6種 + 王将）。
"""

from __future__ import annotations

from enum import IntEnum, unique

ROWS = 9
COLS = 9
NUM_SQUARES = ROWS * COLS  # 81マス


@unique
class Player(IntEnum):
    """Player identifiers.

    先手（SENTE）は下側から上に向かって進む（row 8 → row 0）。
    後手（GOTE）は上側から下に向かって進む（row 0 → row 8）。
    """

    SENTE = 0  # 先手
    GOTE = 1  # 後手

    @property
    def opponent(self) -> Player:
        """相手プレイヤーを返す。"""
        return Player(1 - self.value)


@unique
class PieceType(IntEnum):
    """Piece types in 本将棋（14種類）.

    値は to_tensor_planes() でのチャンネルインデックスに対応する。
    0〜6: 未成駒、7: 王将、8〜13: 成り駒
    """

    PAWN = 0  # 歩
    LANCE = 1  # 香
    KNIGHT = 2  # 桂
    SILVER = 3  # 銀
    GOLD = 4  # 金
    BISHOP = 5  # 角
    ROOK = 6  # 飛
    KING = 7  # 玉/王
    PRO_PAWN = 8  # と（成り歩）
    PRO_LANCE = 9  # 成香
    PRO_KNIGHT = 10  # 成桂
    PRO_SILVER = 11  # 成銀
    HORSE = 12  # 馬（成り角）
    DRAGON = 13  # 龍（成り飛）


# 成り変換テーブル: 未成駒 → 成り駒
PROMOTION_MAP: dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PRO_PAWN,
    PieceType.LANCE: PieceType.PRO_LANCE,
    PieceType.KNIGHT: PieceType.PRO_KNIGHT,
    PieceType.SILVER: PieceType.PRO_SILVER,
    PieceType.BISHOP: PieceType.HORSE,
    PieceType.ROOK: PieceType.DRAGON,
}

# 逆変換: 成り駒 → 元の駒種（取られた駒を持ち駒に戻す際に使用）
UNPROMOTION_MAP: dict[PieceType, PieceType] = {v: k for k, v in PROMOTION_MAP.items()}

# 持ち駒として使える駒種（未成の非玉駒、7種）
HAND_PIECE_TYPES = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
]

# 1マス移動の方向定義（先手視点、前 = 行インデックス減少方向）
# 後手の場合は行方向を反転して使う
STEP_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.PAWN: [(-1, 0)],  # 歩: 1マス前のみ
    PieceType.SILVER: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)],  # 銀: 前3方向+斜め後
    PieceType.GOLD: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],  # 金: 6方向
    PieceType.KING: [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],  # 王: 全8方向1マス
    # 成り駒は金と同じ動き
    PieceType.PRO_PAWN: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_LANCE: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_KNIGHT: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_SILVER: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
}

# 桂馬のジャンプ（2マス前+左右1マス、通常の移動でないため別定義）
KNIGHT_MOVES: list[tuple[int, int]] = [(-2, -1), (-2, 1)]

# 遠距離移動の方向定義（同方向に繰り返し移動できる）
SLIDE_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.LANCE: [(-1, 0)],  # 香: 前方向のみ
    PieceType.BISHOP: [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # 角: 斜め4方向
    PieceType.ROOK: [(-1, 0), (1, 0), (0, -1), (0, 1)],  # 飛: 縦横4方向
    PieceType.HORSE: [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # 馬: 斜め遠距離（+縦横1マス）
    PieceType.DRAGON: [(-1, 0), (1, 0), (0, -1), (0, 1)],  # 龍: 縦横遠距離（+斜め1マス）
}

# 馬（成り角）の追加1マス移動（縦横に1マスだけ動ける）
HORSE_EXTRA_STEPS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 龍（成り飛）の追加1マス移動（斜めに1マスだけ動ける）
DRAGON_EXTRA_STEPS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
