"""Board representation for どうぶつしょうぎ.

盤面のデータ構造。イミュータブル（frozen=True）設計で、
盤面を変更するメソッドはすべて新しい Board オブジェクトを返す。

イミュータブルにする理由:
- 探索木の各ノードが独立した盤面を持てる
- 「元に戻す」操作が不要になる（新しい状態を作るだけ）
- バグが減る（状態が予期せず変わらない）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shogi_ai.game.animal_shogi.types import (
    COLS,
    ROWS,
    PieceType,
    Player,
)


@dataclass(frozen=True)  # イミュータブル（変更不可）なデータクラス
class Piece:
    """A piece on the board.

    盤面上の1つの駒。種類と所有者（先手/後手）を持つ。
    """

    piece_type: PieceType
    owner: Player


@dataclass(frozen=True)
class Board:
    """Immutable board state for 3x4 どうぶつしょうぎ.

    3×4 = 12マスの盤面を表すイミュータブルなデータ構造。

    squares: 12要素のタプル（行優先）。各要素は Piece | None。
             squares[row * COLS + col] でマス(row, col)にアクセス。
    hands: 2要素のタプル。hands[0]=先手の持ち駒、hands[1]=後手の持ち駒。
    """

    squares: tuple[Piece | None, ...] = field(default_factory=lambda: Board._initial_squares())
    hands: tuple[tuple[PieceType, ...], tuple[PieceType, ...]] = ((), ())

    @staticmethod
    def _initial_squares() -> tuple[Piece | None, ...]:
        """Return the standard starting position.

        標準的な初期配置を返す。

        Row 0 (top):    GOTE side — Elephant Lion Giraffe  （後手の陣地）
        Row 1:          _ Chick(GOTE) _
        Row 2:          _ Chick(SENTE) _
        Row 3 (bottom): SENTE side — Giraffe Lion Elephant  （先手の陣地）
        """
        squares: list[Piece | None] = [None] * (ROWS * COLS)

        # 後手の後ろ段（Row 0）: ぞう・ライオン・きりん
        squares[0 * COLS + 0] = Piece(PieceType.GIRAFFE, Player.GOTE)
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[0 * COLS + 2] = Piece(PieceType.ELEPHANT, Player.GOTE)

        # Row 1: 後手のひよこ（中央）
        squares[1 * COLS + 1] = Piece(PieceType.CHICK, Player.GOTE)

        # Row 2: 先手のひよこ（中央）
        squares[2 * COLS + 1] = Piece(PieceType.CHICK, Player.SENTE)

        # 先手の後ろ段（Row 3）: ぞう・ライオン・きりん（左右対称）
        squares[3 * COLS + 0] = Piece(PieceType.ELEPHANT, Player.SENTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        squares[3 * COLS + 2] = Piece(PieceType.GIRAFFE, Player.SENTE)

        return tuple(squares)

    def piece_at(self, row: int, col: int) -> Piece | None:
        """Return the piece at (row, col), or None.

        マス(row, col)の駒を返す。駒がなければ None。
        """
        return self.squares[row * COLS + col]

    def set_piece(self, row: int, col: int, piece: Piece | None) -> Board:
        """Return a new Board with the piece at (row, col) changed.

        マス(row, col)の駒を変更した新しい Board を返す。
        元の Board は変更されない（イミュータブル）。
        """
        idx = row * COLS + col
        squares = list(self.squares)  # タプルをリストに変換して変更
        squares[idx] = piece
        return Board(squares=tuple(squares), hands=self.hands)

    def add_to_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Return a new Board with piece_type added to player's hand.

        プレイヤーの持ち駒に駒を追加した新しい Board を返す。
        成り駒（にわとり）を取ったら元の駒種（ひよこ）に戻す。
        """
        hands = list(self.hands)
        hand = list(hands[player.value])
        # 成り駒を取ったら元に戻す（にわとり → ひよこ）
        if piece_type == PieceType.HEN:
            piece_type = PieceType.CHICK
        hand.append(piece_type)
        hand.sort()  # ソートすることで持ち駒の順序を一意に保つ
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def remove_from_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Return a new Board with one piece_type removed from player's hand.

        プレイヤーの持ち駒から駒を1枚取り除いた新しい Board を返す。
        """
        hands = list(self.hands)
        hand = list(hands[player.value])
        hand.remove(piece_type)  # 最初に見つかった1枚を削除
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def find_lion(self, player: Player) -> int | None:
        """Return the index of player's lion, or None if captured.

        プレイヤーのライオンのマスインデックスを返す。
        ライオンが取られていれば None（勝敗判定に使用）。
        """
        for idx, piece in enumerate(self.squares):
            if piece is not None and piece.piece_type == PieceType.LION and piece.owner == player:
                return idx
        return None
