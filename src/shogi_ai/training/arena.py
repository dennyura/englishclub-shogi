"""Arena for evaluating player strength through head-to-head matches.

アリーナ: 2つのプレイヤー関数を対戦させて強さを評価するモジュール。
AlphaZero では新旧ネットワークを対戦させ、勝率が閾値を超えれば新ネットワークに更新する。
"""

from __future__ import annotations

from collections.abc import Callable

from shogi_ai.game.protocol import GameState


def pit(
    player1_fn: Callable[[GameState], int],
    player2_fn: Callable[[GameState], int],
    initial_state: GameState,
    num_games: int = 50,
    max_moves: int = 200,
) -> tuple[int, int, int]:
    """Play num_games between two players, alternating who goes first.

    2つのプレイヤー関数を num_games 局対戦させる。
    先手・後手を交互に入れ替えることで先手有利バイアスを打ち消す。

    Args:
        player1_fn: 局面を受け取り手を返す関数（プレイヤー1）
        player2_fn: 局面を受け取り手を返す関数（プレイヤー2）
        initial_state: 各局の初期局面
        num_games: 対局数（偶数にすると先後均等になる）
        max_moves: 1局の最大手数（超えたら引き分け扱い）

    Returns:
        (player1_wins, player2_wins, draws)
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for game_idx in range(num_games):
        # 偶数局はプレイヤー1が先手、奇数局はプレイヤー2が先手
        # → 先後有利を均等にする
        if game_idx % 2 == 0:
            sente_fn, gote_fn = player1_fn, player2_fn
            p1_is_sente = True
        else:
            sente_fn, gote_fn = player2_fn, player1_fn
            p1_is_sente = False

        state = initial_state
        move_count = 0

        # 対局ループ
        while not state.is_terminal and move_count < max_moves:
            if state.current_player == 0:  # 先手（SENTE）の番
                move = sente_fn(state)
            else:  # 後手（GOTE）の番
                move = gote_fn(state)
            state = state.apply_move(move)
            move_count += 1

        # 勝敗を判定してプレイヤー1の勝ち負けに変換
        winner = state.winner
        if winner is None or move_count >= max_moves:
            draws += 1  # 引き分けまたは最大手数到達
        elif (winner == 0 and p1_is_sente) or (winner == 1 and not p1_is_sente):
            p1_wins += 1  # プレイヤー1の勝ち
        else:
            p2_wins += 1  # プレイヤー2の勝ち

    return p1_wins, p2_wins, draws
