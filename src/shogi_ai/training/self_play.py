"""Self-play data generation for AlphaZero-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.network import DualHeadNetwork


class TrainingExample(NamedTuple):
    """A single training example from self-play.

    自己対局で生成された1つの訓練データ。

    state_tensor:  局面のテンソル表現（ニューラルネットへの入力）
    policy_target: MCTSの訪問回数から作った目標確率分布（方策の教師）
    value_target:  対局結果（+1=勝, -1=負, 0=引き分け）（価値の教師）
    """

    state_tensor: Tensor  # (in_channels, board_h, board_w)
    policy_target: Tensor  # (action_space_size,)
    value_target: float  # +1 (win) / -1 (loss) / 0 (draw)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    num_games: int = 20
    num_simulations: int = 50
    temperature_threshold: int = 10  # この手数以降は温度を下げて最善手を選ぶ


def play_game(
    network: DualHeadNetwork,
    state: GameState,
    config: SelfPlayConfig,
) -> list[TrainingExample]:
    """Play one game of self-play and return training examples.

    1ゲームの自己対局を行い、訓練データのリストを返す。

    AlphaZero の自己対局プロセス:
    1. 各局面で MCTS を実行して行動確率を得る
    2. 局面・確率・手番プレイヤーを記録する
    3. 対局終了後、各ステップに対局結果（価値）を割り当てる

    Temperature schedule:
    - First `temperature_threshold` moves: τ=1.0 (exploratory)
    - After that: τ→0 (deterministic, pick best)
    """
    examples: list[tuple[Tensor, Tensor, int]] = []
    mcts_config = MCTSConfig(num_simulations=config.num_simulations)
    mcts = MCTS(network, mcts_config)

    move_count = 0
    max_moves = 200  # 無限ループ防止（引き分けとして扱う）

    while not state.is_terminal and move_count < max_moves:
        # 温度スケジュール: 序盤は探索的、中盤以降は最善手を選ぶ
        if move_count < config.temperature_threshold:
            mcts.config = MCTSConfig(
                num_simulations=config.num_simulations,
                temperature=1.0,  # 高温: 多様な手を探索
            )
        else:
            mcts.config = MCTSConfig(
                num_simulations=config.num_simulations,
                temperature=0.01,  # 低温: ほぼ最善手を選択
            )

        # MCTS で行動確率を計算
        action_probs = mcts.search(state)
        tensor = state.to_tensor_planes()
        policy = torch.tensor(action_probs, dtype=torch.float32)

        # (局面テンソル, 方策, 手番プレイヤー) を記録
        # 価値は対局終了後に確定するためここでは記録しない
        examples.append((tensor, policy, state.current_player))

        # 行動確率に従って手を選んで局面を進める
        move = _select_move(action_probs, state.legal_moves())
        state = state.apply_move(move)
        move_count += 1

    # 対局結果が確定したので、各ステップに価値ターゲットを割り当てる
    # 勝ったプレイヤーの手番ステップは +1、負けたら -1、引き分けは 0
    winner = state.winner
    result: list[TrainingExample] = []
    for tensor, policy, player in examples:
        if winner is None:
            value = 0.0  # 引き分け
        elif winner == player:
            value = 1.0  # このプレイヤーが勝った
        else:
            value = -1.0  # このプレイヤーが負けた
        result.append(TrainingExample(tensor, policy, value))

    return result


def generate_training_data(
    network: DualHeadNetwork,
    initial_state: GameState,
    config: SelfPlayConfig,
) -> list[TrainingExample]:
    """Generate training data from multiple self-play games.

    複数の自己対局を行い、訓練データをまとめて返す。
    """
    all_examples: list[TrainingExample] = []
    for _ in range(config.num_games):
        examples = play_game(network, initial_state, config)
        all_examples.extend(examples)
    return all_examples


def _select_move(action_probs: list[float], legal_moves: list[int]) -> int:
    """Sample a move from the action probability distribution.

    行動確率分布に従って手をサンプリングする。
    確率がすべて0の場合は一様分布にフォールバック。
    """
    probs = torch.tensor([action_probs[m] for m in legal_moves])
    # 確率の合計が0の場合は均一分布（フォールバック）
    total = probs.sum()
    if total <= 0:
        idx = torch.randint(len(legal_moves), (1,)).item()
    else:
        probs = probs / total  # 正規化して確率分布に
        idx = torch.multinomial(probs, 1).item()  # 確率に従ってサンプリング
    return legal_moves[int(idx)]
