"""Terminal display for どうぶつしょうぎ boards.

どうぶつしょうぎの盤面をターミナルに表示するためのモジュール。
"""

from __future__ import annotations

from shogi_ai.game.animal_shogi.board import Board
from shogi_ai.game.animal_shogi.types import COLS, ROWS, PieceType, Player

# 駒の表示文字: 大文字=先手、小文字=後手
PIECE_CHARS: dict[PieceType, str] = {
    PieceType.CHICK: "C",  # ひよこ
    PieceType.GIRAFFE: "G",  # きりん
    PieceType.ELEPHANT: "E",  # ぞう
    PieceType.LION: "L",  # ライオン
    PieceType.HEN: "H",  # にわとり（成りひよこ）
}

# 日本語の駒名（CLI や Web UI の表示に使用）
PIECE_NAMES_JA: dict[PieceType, str] = {
    PieceType.CHICK: "ひよこ",
    PieceType.GIRAFFE: "きりん",
    PieceType.ELEPHANT: "ぞう",
    PieceType.LION: "ライオン",
    PieceType.HEN: "にわとり",
}


def piece_to_char(piece_type: PieceType, owner: Player) -> str:
    """Convert a piece to its display character.

    駒を表示文字に変換する。先手は大文字、後手は小文字。
    """
    char = PIECE_CHARS[piece_type]
    if owner == Player.GOTE:
        return char.lower()  # 後手の駒は小文字
    return char  # 先手の駒は大文字


def hand_to_str(hand: tuple[PieceType, ...]) -> str:
    """Convert a hand to display string.

    持ち駒を文字列に変換する。持ち駒なしの場合は "-"。
    """
    if not hand:
        return "-"
    return " ".join(PIECE_CHARS[pt] for pt in hand)


def board_to_str(board: Board) -> str:
    """Convert a board to a human-readable string.

    盤面を人間が読みやすい文字列に変換する。

    Example output:
        GOTE hand: C
          a b c
        1 g l e
        2 . c .
        3 . C .
        4 E L G
        SENTE hand: -

    マス目の見方:
    - 大文字 = 先手の駒
    - 小文字 = 後手の駒
    - "." = 空マス
    - 列ラベル: a, b, c（左から右）
    - 行ラベル: 1, 2, 3, 4（上から下）
    """
    lines: list[str] = []

    # 後手の持ち駒（上段に表示）
    lines.append(f"GOTE hand: {hand_to_str(board.hands[Player.GOTE.value])}")

    # 列ヘッダー（a, b, c）
    col_labels = " ".join(chr(ord("a") + c) for c in range(COLS))
    lines.append(f"  {col_labels}")

    # 盤面の各行
    for r in range(ROWS):
        row_chars: list[str] = []
        for c in range(COLS):
            piece = board.piece_at(r, c)
            if piece is None:
                row_chars.append(".")
            else:
                row_chars.append(piece_to_char(piece.piece_type, piece.owner))
        lines.append(f"{r + 1} {' '.join(row_chars)}")

    # 先手の持ち駒（下段に表示）
    lines.append(f"SENTE hand: {hand_to_str(board.hands[Player.SENTE.value])}")

    return "\n".join(lines)
