"""Board representation for 本将棋 (9x9).

9×9盤の盤面データ構造。どうぶつしょうぎの Board と同じ設計方針：
イミュータブルなデータクラスで、変更メソッドは新しいオブジェクトを返す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shogi_ai.game.full_shogi.types import (
    COLS,
    NUM_SQUARES,
    ROWS,
    UNPROMOTION_MAP,
    PieceType,
    Player,
)


@dataclass(frozen=True)
class Piece:
    """A piece on the board.

    盤面上の駒。種類と所有者を持つ。
    """

    piece_type: PieceType
    owner: Player


@dataclass(frozen=True)
class Board:
    """Immutable board state for 9x9 本将棋.

    9×9 = 81マスの盤面を表すイミュータブルなデータ構造。

    squares: 81要素のタプル（行優先）。squares[row * COLS + col] でアクセス。
    hands: 2要素のタプル。hands[0]=先手の持ち駒、hands[1]=後手の持ち駒。
    """

    squares: tuple[Piece | None, ...] = field(default_factory=lambda: Board._initial_squares())
    hands: tuple[tuple[PieceType, ...], tuple[PieceType, ...]] = ((), ())

    @staticmethod
    def _initial_squares() -> tuple[Piece | None, ...]:
        """Return the standard starting position (平手).

        本将棋の標準初期配置（平手）を返す。

        Row 0 = 後手の後段（上端）、Row 8 = 先手の後段（下端）。
        将棋盤の「9筋」表記と異なり、プログラムでは列0が左（9筋）になる点に注意。
        """
        squares: list[Piece | None] = [None] * NUM_SQUARES

        # 後手の後段（Row 0）: 香桂銀金王金銀桂香
        gote_back = [
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.KING,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
        ]
        for c, pt in enumerate(gote_back):
            squares[0 * COLS + c] = Piece(pt, Player.GOTE)

        # Row 1: 後手の飛角（飛車=右、角行=左に配置）
        squares[1 * COLS + 1] = Piece(PieceType.ROOK, Player.GOTE)  # 飛車
        squares[1 * COLS + 7] = Piece(PieceType.BISHOP, Player.GOTE)  # 角行

        # Row 2: 後手の歩兵（9枚）
        for c in range(COLS):
            squares[2 * COLS + c] = Piece(PieceType.PAWN, Player.GOTE)

        # 先手の歩兵（Row 6）
        for c in range(COLS):
            squares[6 * COLS + c] = Piece(PieceType.PAWN, Player.SENTE)

        # Row 7: 先手の飛角（後手と鏡像）
        squares[7 * COLS + 1] = Piece(PieceType.BISHOP, Player.SENTE)  # 角行
        squares[7 * COLS + 7] = Piece(PieceType.ROOK, Player.SENTE)  # 飛車

        # 先手の後段（Row 8）: 香桂銀金王金銀桂香
        sente_back = [
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.KING,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
        ]
        for c, pt in enumerate(sente_back):
            squares[8 * COLS + c] = Piece(pt, Player.SENTE)

        return tuple(squares)

    def piece_at(self, row: int, col: int) -> Piece | None:
        """マス(row, col)の駒を返す。駒がなければ None。"""
        return self.squares[row * COLS + col]

    def set_piece(self, row: int, col: int, piece: Piece | None) -> Board:
        """マス(row, col)の駒を変更した新しい Board を返す。"""
        idx = row * COLS + col
        squares = list(self.squares)
        squares[idx] = piece
        return Board(squares=tuple(squares), hands=self.hands)

    def add_to_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Add piece to hand, reverting promoted pieces to base form.

        取った駒を持ち駒に追加する。成り駒は元の駒種に戻す。
        例: 龍（成り飛）を取ったら、飛車として持ち駒に加える。
        """
        hands = list(self.hands)
        hand = list(hands[player.value])
        # 成り駒を取ったら元に戻す（UNPROMOTION_MAP で逆引き）
        base_type = UNPROMOTION_MAP.get(piece_type, piece_type)
        hand.append(base_type)
        hand.sort()  # 一意な順序を保つ
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def remove_from_hand(self, player: Player, piece_type: PieceType) -> Board:
        """持ち駒から1枚取り除いた新しい Board を返す。"""
        hands = list(self.hands)
        hand = list(hands[player.value])
        hand.remove(piece_type)
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def find_king(self, player: Player) -> int | None:
        """プレイヤーの王将のマスインデックスを返す。王将がなければ None。

        チェック判定や終局判定に使用する。
        """
        for idx, piece in enumerate(self.squares):
            if piece is not None and piece.piece_type == PieceType.KING and piece.owner == player:
                return idx
        return None

    def count_pawns_in_column(self, player: Player, col: int) -> int:
        """Count unpromoted pawns of player in a column (for 二歩 check).

        指定列にあるプレイヤーの未成歩の枚数を返す。
        二歩（同じ列に2枚の歩を置くこと）を判定するために使用する。
        """
        count = 0
        for r in range(ROWS):
            p = self.piece_at(r, col)
            if p is not None and p.owner == player and p.piece_type == PieceType.PAWN:
                count += 1
        return count
