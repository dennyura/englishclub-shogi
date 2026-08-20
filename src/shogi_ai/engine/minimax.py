"""Minimax search with alpha-beta pruning for どうぶつしょうぎ."""

from __future__ import annotations

from shogi_ai.game.protocol import GameState

# 駒の価値テーブル（材料評価に使用）
# ライオンに高い値を設定することで「ライオンを守る」行動を優先させる
_PIECE_VALUES = {
    0: 1.0,  # CHICK（ひよこ）
    1: 3.0,  # GIRAFFE（きりん）
    2: 3.0,  # ELEPHANT（ぞう）
    3: 100.0,  # LION（ライオン）— 圧倒的に高い値でライオン保護を最優先
    4: 5.0,  # HEN（にわとり、成りひよこ）
}


def evaluate(state: GameState) -> float:
    """Evaluate a position from the current player's perspective.

    局面を現在のプレイヤーの視点から数値評価する（静的評価関数）。

    Scoring:
    - Material advantage (piece values)
    - Terminal bonus/penalty (±1000)

    Returns positive if current player is better off.
    """
    if state.is_terminal:
        if state.winner is None:
            return 0.0  # 引き分け
        if state.winner == state.current_player:
            return 1000.0  # 勝ち
        return -1000.0  # 負け

    # 盤上の駒を数えて材料差を計算
    # 現在のプレイヤーの駒は +値、相手の駒は -値
    score = 0.0
    board = state.board  # type: ignore[attr-defined]

    # 盤上の駒を評価
    for piece in board.squares:
        if piece is None:
            continue
        value = _PIECE_VALUES.get(piece.piece_type.value, 0.0)
        if piece.owner.value == state.current_player:
            score += value  # 自分の駒
        else:
            score -= value  # 相手の駒

    # 持ち駒も評価（持ち駒は潜在的な打ち駒として価値がある）
    for i, hand in enumerate(board.hands):
        for pt in hand:
            value = _PIECE_VALUES.get(pt.value, 0.0)
            if i == state.current_player:
                score += value
            else:
                score -= value

    return score


def negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
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
    # 終局状態の評価
    if state.is_terminal:
        if state.winner is None:
            return -1, 0.0
        if state.winner == state.current_player:
            # depth を加算することで「より速い勝利」を優先する
            return -1, 1000.0 + depth
        return -1, -(1000.0 + depth)

    # 探索深さ0に達したら静的評価を返す（葉ノード）
    if depth == 0:
        return -1, evaluate(state)

    moves = state.legal_moves()
    best_move = moves[0]
    best_score = float("-inf")

    for move in moves:
        next_state = state.apply_move(move)
        # 相手番の評価値を符号反転して自分の視点に変換（ネガマックスの核心）
        _, score = negamax(next_state, depth - 1, -beta, -alpha)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        # α値を更新（自分が保証できる最低スコアを引き上げる）
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # βカットオフ: 相手はこの枝を選ばないので探索打ち切り

    return best_move, best_score


def minimax_move(state: GameState, depth: int = 4) -> int:
    """Return the best move for the current player using minimax search.

    ミニマックス探索で最善手を返す。
    depth=4 はどうぶつしょうぎ向けのデフォルト値。
    本将棋では組み合わせ爆発を避けるため depth=2 程度に抑える。
    """
    move, _ = negamax(state, depth, float("-inf"), float("inf"))
    return move
