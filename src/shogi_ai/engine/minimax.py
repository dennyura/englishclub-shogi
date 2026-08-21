"""Minimax search with alpha-beta pruning for どうぶつしょうぎ."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, cast

from shogi_ai.game.full_shogi.moves import decode_move as decode_full_shogi_move
from shogi_ai.game.full_shogi.types import UNPROMOTION_MAP
from shogi_ai.game.full_shogi.types import PieceType as FullShogiPieceType
from shogi_ai.game.protocol import GameState

TTFlag = Literal["exact", "lower", "upper"]


@dataclass(frozen=True)
class _TTEntry:
    """Cached negamax result and its alpha-beta bound type."""

    move: int
    score: float
    flag: TTFlag


_TranspositionTable = dict[tuple[Hashable, int, int], _TTEntry]

# 駒の価値テーブル（材料評価に使用）
# ライオンに高い値を設定することで「ライオンを守る」行動を優先させる
_PIECE_VALUES = {
    0: 1.0,  # CHICK（ひよこ）
    1: 3.0,  # GIRAFFE（きりん）
    2: 3.0,  # ELEPHANT（ぞう）
    3: 100.0,  # LION（ライオン）— 圧倒的に高い値でライオン保護を最優先
    4: 5.0,  # HEN（にわとり、成りひよこ）
}

# 本将棋の未成駒の価値。
_FULL_SHOGI_BASE_VALUE = {
    FullShogiPieceType.PAWN: 200.0,  # P
    FullShogiPieceType.LANCE: 430.0,  # L
    FullShogiPieceType.KNIGHT: 450.0,  # N
    FullShogiPieceType.SILVER: 640.0,  # S
    FullShogiPieceType.GOLD: 730.0,  # G
    FullShogiPieceType.BISHOP: 1040.0,  # B
    FullShogiPieceType.ROOK: 1040.0,  # R
    FullShogiPieceType.KING: 10000.0,  # K
}

# 本将棋の成りによる追加価値。キーは成る前の駒種。
_FULL_SHOGI_PROMOTE_BONUS = {
    FullShogiPieceType.PAWN: 320.0,  # P
    FullShogiPieceType.LANCE: 200.0,  # L
    FullShogiPieceType.KNIGHT: 190.0,  # N
    FullShogiPieceType.SILVER: 30.0,  # S
    FullShogiPieceType.GOLD: 0.0,  # G
    FullShogiPieceType.BISHOP: 260.0,  # B
    FullShogiPieceType.ROOK: 260.0,  # R
}

# 本将棋の持ち駒の価値。キーは持ち駒の駒種（常に未成）。
_FULL_SHOGI_HAND_BONUS = {
    FullShogiPieceType.PAWN: 15.0,  # P
    FullShogiPieceType.LANCE: 50.0,  # L
    FullShogiPieceType.KNIGHT: 60.0,  # N
    FullShogiPieceType.SILVER: 80.0,  # S
    FullShogiPieceType.GOLD: 90.0,  # G
    FullShogiPieceType.BISHOP: 230.0,  # B
    FullShogiPieceType.ROOK: 230.0,  # R
}


def _full_shogi_piece_value(piece_type: FullShogiPieceType, in_hand: bool) -> float:
    """Return the full-shogi value for a board or hand piece."""
    if in_hand:
        return _FULL_SHOGI_BASE_VALUE.get(piece_type, 0.0) + _FULL_SHOGI_HAND_BONUS.get(
            piece_type, 0.0
        )
    if piece_type in _FULL_SHOGI_BASE_VALUE:
        return _FULL_SHOGI_BASE_VALUE[piece_type]
    if piece_type in UNPROMOTION_MAP:
        base_piece = UNPROMOTION_MAP[piece_type]
        return _FULL_SHOGI_BASE_VALUE[base_piece] + _FULL_SHOGI_PROMOTE_BONUS[base_piece]
    return 0.0


def _piece_value(piece_type: object, in_hand: bool = False) -> float:
    """Return the configured value for an animal- or full-shogi piece."""
    if isinstance(piece_type, FullShogiPieceType):
        return _full_shogi_piece_value(piece_type, in_hand)
    return _PIECE_VALUES.get(piece_type.value, 0.0)  # type: ignore[attr-defined]


def evaluate(state: GameState) -> float:
    """Evaluate a position from the current player's perspective.

    局面を現在のプレイヤーの視点から数値評価する（静的評価関数）。

    Scoring:
    - Material advantage (piece values)
    - Terminal bonus/penalty (±10000)

    Returns positive if current player is better off.
    """
    if state.is_terminal:
        if state.winner is None:
            return 0.0  # 引き分け
        if state.winner == state.current_player:
            return 10000.0  # 勝ち
        return -10000.0  # 負け

    # 盤上の駒を数えて材料差を計算
    # 現在のプレイヤーの駒は +値、相手の駒は -値
    score = 0.0
    board = state.board  # type: ignore[attr-defined]

    # 盤上の駒を評価
    for piece in board.squares:
        if piece is None:
            continue
        value = _piece_value(piece.piece_type)
        if piece.owner.value == state.current_player:
            score += value  # 自分の駒
        else:
            score -= value  # 相手の駒

    # 持ち駒も評価（持ち駒は潜在的な打ち駒として価値がある）
    for i, hand in enumerate(board.hands):
        for pt in hand:
            value = _piece_value(pt, in_hand=True)
            if i == state.current_player:
                score += value
            else:
                score -= value

    return score


def _mvv_lva_score(state: GameState, move: int) -> float:
    """Return an MVV-LVA ordering score for a full-shogi board move."""
    board = getattr(state, "board", None)
    if board is None or len(board.squares) != 81:
        return 0.0

    decoded = decode_full_shogi_move(move)
    if decoded["type"] != "board":
        return 0.0

    from_row, from_col = decoded["from"]
    to_row, to_col = decoded["to"]
    attacker = board.piece_at(from_row, from_col)
    victim = board.piece_at(to_row, to_col)
    score = 0.0

    if attacker is not None and victim is not None and attacker.owner != victim.owner:
        attacker_value = _full_shogi_piece_value(attacker.piece_type, in_hand=False)
        victim_value = _full_shogi_piece_value(victim.piece_type, in_hand=False)
        # MVV-LVA: first maximize the captured value, then minimize the attacker value.
        score += 1_000_000.0 + victim_value * 100.0 - attacker_value

    if decoded["promote"]:
        score += 10_000.0

    return score


def _order_moves(state: GameState, moves: list[int], preferred_move: int) -> list[int]:
    """Order moves using the transposition-table move and MVV-LVA."""
    if preferred_move in moves:
        moves.remove(preferred_move)
        moves.insert(0, preferred_move)
        remaining = moves[1:]
    else:
        remaining = moves
    remaining.sort(key=lambda move: _mvv_lva_score(state, move), reverse=True)
    return moves[:1] + remaining if preferred_move in moves else remaining


def negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    _table: _TranspositionTable | None = None,
) -> tuple[int, float]:
    """Negamax search with alpha-beta pruning.

    ネガマックス法 + αβ枝刈りによる探索。

    ネガマックス法とは:
    ミニマックス法の変形で、常に「現在のプレイヤーにとっての評価値」を
    返すようにする。相手番の評価値は符号を反転させることで統一できる。

    αβ枝刈りとは:
    探索不要な枝を切り捨て、ミニマックスと同じ結果をより速く得る手法。
    alpha: 現在のプレイヤーが保証できる最低スコア
    beta:  相手のプレイヤーが保証できる最低スコア（現在プレイヤーにとっての上限）

    Returns (best_move, score) from the current player's perspective.
    best_move is -1 when depth=0 or at terminal states.
    """
    if _table is None:
        _table = {}

    original_alpha = alpha
    original_beta = beta
    board = cast(Hashable, getattr(state, "board", state))
    position_key = (board, state.current_player, depth)
    cached = _table.get(position_key)
    if cached is not None:
        if cached.flag == "exact":
            return cached.move, cached.score
        if cached.flag == "lower":
            alpha = max(alpha, cached.score)
        else:
            beta = min(beta, cached.score)
        if alpha >= beta:
            return cached.move, cached.score

    # 終局状態の評価
    if state.is_terminal:
        if state.winner is None:
            result = -1, 0.0
        elif state.winner == state.current_player:
            # depth を加算することで「より速い勝利」を優先する
            result = -1, 20000.0 + depth
        else:
            result = -1, -(20000.0 + depth)
        _table[position_key] = _TTEntry(*result, "exact")
        return result

    # 探索深さ0に達したら静的評価を返す（葉ノード）
    if depth == 0:
        result = -1, evaluate(state)
        _table[position_key] = _TTEntry(*result, "exact")
        return result

    moves = _order_moves(state, state.legal_moves(), cached.move if cached is not None else -1)
    best_move = moves[0]
    best_score = float("-inf")

    for move in moves:
        next_state = state.apply_move(move)
        # 相手番の評価値を符号反転して自分の視点に変換（ネガマックスの核心）
        _, score = negamax(next_state, depth - 1, -beta, -alpha, _table)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        # α値を更新（自分が保証できる最低スコアを引き上げる）
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # βカットオフ: 相手はこの枝を選ばないので探索打ち切り

    if best_score <= original_alpha:
        flag: TTFlag = "upper"
    elif best_score >= original_beta:
        flag = "lower"
    else:
        flag = "exact"
    _table[position_key] = _TTEntry(best_move, best_score, flag)
    return best_move, best_score


def minimax_scores(
    state: GameState,
    depth: int = 4,
) -> list[tuple[int, float]]:
    """Return every legal move and its Minimax score in descending order."""
    if depth < 1:
        raise ValueError("depth must be at least 1")

    table: _TranspositionTable = {}
    scored_moves: list[tuple[int, float]] = []

    for move in state.legal_moves():
        next_state = state.apply_move(move)
        _, score = negamax(
            next_state,
            depth - 1,
            float("-inf"),
            float("inf"),
            table,
        )
        scored_moves.append((move, -score))

    return sorted(scored_moves, key=lambda item: item[1], reverse=True)


def minimax_move(state: GameState, depth: int = 4) -> int:
    """Return the best move for the current player using minimax search.

    ミニマックス探索で最善手を返す。
    depth=4 はどうぶつしょうぎ向けのデフォルト値。
    本将棋では組み合わせ爆発を避けるため depth=2 程度に抑える。
    """
    move, _ = negamax(state, depth, float("-inf"), float("inf"))
    return move
