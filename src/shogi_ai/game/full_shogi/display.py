"""Terminal display for 本将棋."""

from __future__ import annotations

from shogi_ai.game.full_shogi.board import Board
from shogi_ai.game.full_shogi.types import COLS, ROWS, PieceType, Player

# Display characters for pieces
_PIECE_CHARS: dict[PieceType, str] = {
    PieceType.PAWN: "歩",
    PieceType.LANCE: "香",
    PieceType.KNIGHT: "桂",
    PieceType.SILVER: "銀",
    PieceType.GOLD: "金",
    PieceType.BISHOP: "角",
    PieceType.ROOK: "飛",
    PieceType.KING: "玉",
    PieceType.PRO_PAWN: "と",
    PieceType.PRO_LANCE: "杏",
    PieceType.PRO_KNIGHT: "圭",
    PieceType.PRO_SILVER: "全",
    PieceType.HORSE: "馬",
    PieceType.DRAGON: "龍",
}


def format_board(board: Board) -> str:
    """Format the board for terminal display."""
    lines: list[str] = []

    # Gote's hand
    gote_hand = _format_hand(board, Player.GOTE)
    lines.append(f"後手持駒: {gote_hand}")
    lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
    lines.append("+--+--+--+--+--+--+--+--+--+")

    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            piece = board.piece_at(r, c)
            if piece is None:
                row_str += "  |"
            else:
                char = _PIECE_CHARS.get(piece.piece_type, "？")
                if piece.owner == Player.GOTE:
                    row_str += f"v{char[0]}|"
                else:
                    row_str += f" {char[0]}|"
        lines.append(f"{row_str} {_row_label(r)}")
        lines.append("+--+--+--+--+--+--+--+--+--+")

    # Sente's hand
    sente_hand = _format_hand(board, Player.SENTE)
    lines.append(f"先手持駒: {sente_hand}")

    return "\n".join(lines)


def _format_hand(board: Board, player: Player) -> str:
    hand = board.hands[player.value]
    if not hand:
        return "なし"
    pieces: list[str] = []
    for pt in sorted(set(hand)):
        count = hand.count(pt)
        char = _PIECE_CHARS.get(pt, "？")
        if count == 1:
            pieces.append(char)
        else:
            pieces.append(f"{char}{count}")
    return " ".join(pieces)


def _row_label(row: int) -> str:
    labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    return labels[row]
