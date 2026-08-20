"""Legal move generation for どうぶつしょうぎ (円筒形/トーラス将棋 variant).

CYLINDRICAL BOARD RULE:
  The left and right edges of the board are connected (column wraps around).
  A piece moving off the rightmost column continues from column 0, and vice
  versa. Rows are NOT connected — top/bottom boundary rules (promotion,
  falling off the board) are unchanged from standard どうぶつしょうぎ.

  Since every piece in animal shogi moves exactly one square (there are no
  sliding pieces), the only change needed is: destination column is computed
  as `(col + dc) % COLS` instead of being bounds-checked and rejected when
  it falls outside [0, COLS). Destination row is still bounds-checked
  normally with `0 <= nr < ROWS`.

合法手の生成と手のエンコード/デコード。

手のエンコード（整数値への変換）:
  盤上の手: from_idx * 12 + to_idx     (0〜143)
    from_idx: 移動元マスのインデックス（0〜11）
    to_idx:   移動先マスのインデックス（0〜11）
  打ち手:   144 + piece_type * 12 + to_idx  (144〜179)
    piece_type: 0=ひよこ, 1=きりん, 2=ぞう（打てる3種）

  合計行動空間: 180
"""

from __future__ import annotations

from typing import Final

from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    PIECE_MOVES,
    ROWS,
    PieceType,
    Player,
)

ACTION_SPACE = 180  # どうぶつしょうぎの全行動数
# 盤上の手の総数（from_idx × to_idx: 12 × 12 = 144）
# DROP_OFFSET 以降の値が打ち駒手（144〜179）に対応する
DROP_OFFSET: Final[int] = ROWS * COLS * ROWS * COLS  # = 144


def encode_board_move(from_idx: int, to_idx: int) -> int:
    """Encode a board move as an integer.

    盤上の手を整数にエンコードする。
    from_idx * 12 + to_idx（12 は盤面のマス数）
    """
    return from_idx * 12 + to_idx


def encode_drop_move(piece_type: PieceType, to_idx: int) -> int:
    """Encode a drop move as an integer.

    持ち駒打ちを整数にエンコードする。
    DROP_OFFSET（盤上の手の最大値+1）以降の値を使う。
    """
    pt_index = HAND_PIECE_TYPES.index(piece_type)
    return DROP_OFFSET + pt_index * 12 + to_idx


def decode_move(move: int) -> dict:
    """Decode a move integer into a descriptive dict.

    整数の手を人間が読める辞書形式にデコードする。
    表示や Web API のレスポンスで使用する。
    """
    if move < DROP_OFFSET:  # 盤上の手
        from_idx = move // 12
        to_idx = move % 12
        return {
            "type": "board",
            "from": (from_idx // COLS, from_idx % COLS),
            "to": (to_idx // COLS, to_idx % COLS),
        }
    else:  # 打ち手
        remainder = move - DROP_OFFSET
        pt_index = remainder // 12
        to_idx = remainder % 12
        return {
            "type": "drop",
            "piece_type": HAND_PIECE_TYPES[pt_index],
            "to": (to_idx // COLS, to_idx % COLS),
        }


def legal_moves(board: Board, player: Player) -> list[int]:
    """Generate all legal moves for the given player.

    プレイヤーのすべての合法手を生成する。

    Includes:
    - Board moves (piece movement + promotion)（盤上の手・成り）
    - Drop moves (placing captured pieces)（持ち駒打ち）

    Does NOT include moves that leave own lion in check
    (lion capture is allowed as a win condition).
    自玉が取られる手は除外しない（ライオン取りが勝利条件のため）。
    """
    moves: list[int] = []

    # --- 盤上の手の生成 ---
    for idx, piece in enumerate(board.squares):
        if piece is None or piece.owner != player:
            continue  # 空マスまたは相手の駒はスキップ
        row, col = idx // COLS, idx % COLS
        deltas = PIECE_MOVES[piece.piece_type]

        for dr, dc in deltas:
            # 後手（GOTE）は移動方向を縦に反転（先後対称な設計）
            if player == Player.GOTE:
                dr = -dr
            nr = row + dr
            nc = (col + dc) % COLS  # 円筒ラップ（左端⇔右端がつながる）
            if not (0 <= nr < ROWS):
                continue  # 上下方向の盤外はスキップ（列は巻き付くので判定不要）
            target = board.piece_at(nr, nc)
            if target is not None and target.owner == player:
                continue  # 自分の駒のある場所には動けない

            to_idx = nr * COLS + nc
            move = encode_board_move(idx, to_idx)

            # 成り判定（ひよこが相手陣地後段に到達 → 自動的ににわとりになる）
            if _should_promote(piece, player, nr):
                moves.append(move)  # どうぶつしょうぎの成りは強制・任意同一
            else:
                moves.append(move)

    # --- 持ち駒打ちの生成 ---
    hand = board.hands[player.value]
    unique_in_hand = set(hand)  # 同じ駒種を重複して生成しないよう集合に
    for pt in unique_in_hand:
        for idx in range(ROWS * COLS):
            if board.squares[idx] is not None:
                continue  # 駒のあるマスには打てない
            moves.append(encode_drop_move(pt, idx))

    return moves


def apply_move(board: Board, player: Player, move: int) -> Board:
    """Apply a move and return the new board state.

    手を適用して新しい盤面を返す。
    整数の手を見て盤上の手か打ち手かを判別する。
    """
    if move < DROP_OFFSET:
        return _apply_board_move(board, player, move)
    else:
        return _apply_drop_move(board, player, move)


def _apply_board_move(board: Board, player: Player, move: int) -> Board:
    """Apply a board move (piece movement).

    盤上の手を適用する（駒を移動させる）。
    """
    from_idx = move // 12
    to_idx = move % 12
    from_row, from_col = from_idx // COLS, from_idx % COLS
    to_row = to_idx // COLS

    piece = board.piece_at(from_row, from_col)
    assert piece is not None, f"No piece at ({from_row}, {from_col})"

    # 駒を取る処理（相手の駒があれば持ち駒に加える）
    target = board.squares[to_idx]
    new_board = board
    if target is not None:
        new_board = new_board.add_to_hand(player, target.piece_type)

    # 成り判定: ひよこが相手後段に到達したらにわとりに成る
    new_piece_type = piece.piece_type
    if _should_promote(piece, player, to_row):
        new_piece_type = PieceType.HEN

    # 駒を移動: 移動元を空にして、移動先に新しい駒を置く
    new_board = new_board.set_piece(from_row, from_col, None)
    new_board = new_board.set_piece(
        to_idx // COLS,
        to_idx % COLS,
        Piece(new_piece_type, player),
    )

    return new_board


def _apply_drop_move(board: Board, player: Player, move: int) -> Board:
    """Apply a drop move (placing a captured piece).

    持ち駒打ちを適用する（持ち駒を盤上に置く）。
    """
    remainder = move - DROP_OFFSET
    pt_index = remainder // 12
    to_idx = remainder % 12

    piece_type = HAND_PIECE_TYPES[pt_index]
    to_row, to_col = to_idx // COLS, to_idx % COLS

    # 持ち駒から1枚取り除いて盤上に配置
    new_board = board.remove_from_hand(player, piece_type)
    new_board = new_board.set_piece(to_row, to_col, Piece(piece_type, player))

    return new_board


def _should_promote(piece: Piece, player: Player, dest_row: int) -> bool:
    """Check if a piece should promote on reaching dest_row.

    駒が成るべきかどうかを判定する。
    どうぶつしょうぎでは、ひよこが相手陣地後段に到達したら必ずにわとりになる。
    """
    if piece.piece_type != PieceType.CHICK:
        return False  # ひよこ以外は成れない
    if player == Player.SENTE:
        return dest_row == 0  # 先手は row 0（盤面上端）で成る
    else:
        return dest_row == ROWS - 1  # 後手は row 3（盤面下端）で成る