"""CLI entry point for shogi-ai — Human vs Random AI.

コマンドラインで動くどうぶつしょうぎ対局プログラム。
プレイヤー（先手）対ランダムAI（後手）で対局できる。

起動方法: `uv run shogi-cli`
"""

from __future__ import annotations

from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.display import (
    PIECE_NAMES_JA,
    board_to_str,
)
from shogi_ai.game.animal_shogi.moves import decode_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import Player


def _format_move(move: int) -> str:
    """Format a move for display.

    手を人間が読みやすい文字列に変換する。
    例: 盤上の手 → "b2 -> b3"
        持ち駒打ち → "drop ひよこ -> b3"
    """
    info = decode_move(move)
    if info["type"] == "board":
        fr, fc = info["from"]
        tr, tc = info["to"]
        # 列: a/b/c（アルファベット）、行: 1/2/3/4（数字）
        return f"{chr(ord('a') + fc)}{fr + 1} -> {chr(ord('a') + tc)}{tr + 1}"
    else:
        pt = info["piece_type"]
        tr, tc = info["to"]
        return f"drop {PIECE_NAMES_JA[pt]} -> {chr(ord('a') + tc)}{tr + 1}"


def main() -> None:
    """Run a Human (SENTE) vs Random AI (GOTE) game.

    人間（先手）対ランダムAI（後手）の対局を実行する。

    ゲームの流れ:
    1. 盤面を表示
    2. 合法手一覧を表示して番号入力を求める
    3. AI が応答する
    4. 終局まで繰り返す
    """
    print("=== どうぶつしょうぎ ===")
    print("You are SENTE (uppercase). AI is GOTE (lowercase).")
    print()

    state = AnimalShogiState()  # 初期局面から開始

    while not state.is_terminal:
        print(board_to_str(state.board))
        print()

        if state.current_player == Player.SENTE:
            # プレイヤー（先手）の番
            moves = state.legal_moves()
            print("Legal moves:")
            for i, m in enumerate(moves):
                print(f"  {i}: {_format_move(m)}")
            print()

            # 入力検証ループ（正しい番号が入力されるまで繰り返す）
            while True:
                try:
                    choice = input("Your move (number): ")
                    idx = int(choice)
                    if 0 <= idx < len(moves):
                        state = state.apply_move(moves[idx])
                        break
                    print(f"Invalid: choose 0-{len(moves) - 1}")
                except ValueError:
                    print("Enter a number.")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return
        else:
            # AI（後手）の番: ランダムに手を選んで指す
            move = random_move(state)
            print(f"AI plays: {_format_move(move)}")
            state = state.apply_move(move)

        print()

    # 終局: 結果を表示
    print(board_to_str(state.board))
    print()
    winner = state.winner
    if winner == Player.SENTE:
        print("You win!")
    elif winner == Player.GOTE:
        print("AI wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()
