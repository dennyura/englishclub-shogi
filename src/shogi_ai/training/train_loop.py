"""AlphaZero 訓練ループ — 自己対局 → 訓練 → アリーナ評価 → 採用判定 → 保存.

AlphaZeroの学習は次のサイクルを繰り返す:
  1. 自己対局: 現在の最良ネットワークで対局してデータを生成する
  2. 訓練: 生成データでネットワークを強化する
  3. アリーナ: 新旧ネットワークを対戦させて強くなったか確認する（相対評価）
  4. 採用: 勝率が閾値を超えれば新ネットワークを採用してモデルを保存する

「ランダムAIに勝てるか」という絶対評価ではなく、「前世代より強いか」という相対評価を
使うことで、どうぶつしょうぎでも本将棋でも1世代ごとに改善を確認できる。
"""

from __future__ import annotations

import copy
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import NetworkConfig
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.arena import pit
from shogi_ai.training.self_play import SelfPlayConfig, generate_training_data
from shogi_ai.training.trainer import Trainer, TrainerConfig


@dataclass(frozen=True)
class TrainLoopConfig:
    """訓練ループの設定パラメータ。

    Attributes:
        num_generations:      世代数（1世代 = 自己対局+訓練+アリーナ）
        num_self_play_games:  1世代あたりの自己対局数
        num_simulations:      MCTSシミュレーション数（多いほど強いが遅い）
        arena_games:          アリーナ対戦数（新旧比較、偶数推奨）
        win_rate_threshold:   新モデル採用の勝率閾値（55%以上で採用）
        model_path:           最良モデルの保存先パス
    """

    num_generations: int = 10
    num_self_play_games: int = 5
    num_simulations: int = 25
    arena_games: int = 10
    win_rate_threshold: float = 0.55
    model_path: str = "best_model.pt"


def _get_device() -> torch.device:
    """利用可能なデバイスを返す（MPS > CUDA > CPU）."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_mcts_fn(
    network: DualHeadNetwork,
    num_simulations: int,
) -> Callable[[GameState], int]:
    """MCTS手選択関数を作成する。アリーナ対戦・対局で使用。

    temperature=0.01 にすることで、ほぼ最善手を選ぶ確定的な行動になる。
    """
    mcts = MCTS(network, MCTSConfig(num_simulations=num_simulations, temperature=0.01))

    def fn(state: GameState) -> int:
        probs = mcts.search(state)
        legal = state.legal_moves()
        return max(legal, key=lambda m: probs[m])

    return fn


def run_training(
    initial_state: GameState,
    network_config: NetworkConfig,
    loop_config: TrainLoopConfig,
    progress_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    """訓練ループ本体。バックグラウンドスレッドで実行される。

    各世代で以下を実行:
      1. 自己対局でデータ生成
      2. 新ネットワークを訓練
      3. アリーナで新旧対戦（相対評価）
      4. 勝率が閾値以上なら採用・モデル保存

    進捗は progress_queue に dict を入れて Web UI（SSE）に伝える。
    stop_event がセットされれば途中で安全に終了する。
    """
    device = _get_device()

    # 最良モデルを初期化（または保存済みモデルから続きを再開）
    best_network = DualHeadNetwork(network_config).to(device)
    model_path = Path(loop_config.model_path)
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        best_network.load_state_dict(state_dict)

    trainer_config = TrainerConfig()
    self_play_config = SelfPlayConfig(
        num_games=loop_config.num_self_play_games,
        num_simulations=loop_config.num_simulations,
    )

    for generation in range(loop_config.num_generations):
        if stop_event.is_set():
            progress_queue.put({"type": "stopped"})
            return

        # ── Phase 1: 自己対局 ──────────────────────────────────────────
        progress_queue.put(
            {
                "type": "phase",
                "generation": generation + 1,
                "total": loop_config.num_generations,
                "phase": "self_play",
            }
        )

        best_network.eval()
        data = generate_training_data(best_network, initial_state, self_play_config)

        if stop_event.is_set():
            progress_queue.put({"type": "stopped"})
            return

        # ── Phase 2: 訓練 ──────────────────────────────────────────────
        progress_queue.put(
            {
                "type": "phase",
                "generation": generation + 1,
                "total": loop_config.num_generations,
                "phase": "training",
                "data_size": len(data),
            }
        )

        new_network = copy.deepcopy(best_network).to(device)
        trainer = Trainer(new_network, trainer_config, device)
        losses = trainer.train(data)

        if stop_event.is_set():
            progress_queue.put({"type": "stopped"})
            return

        # ── Phase 3: アリーナ対戦（新旧比較） ─────────────────────────
        progress_queue.put(
            {
                "type": "phase",
                "generation": generation + 1,
                "total": loop_config.num_generations,
                "phase": "arena",
            }
        )

        new_network.eval()
        best_network.eval()
        new_fn = _make_mcts_fn(new_network, loop_config.num_simulations)
        old_fn = _make_mcts_fn(best_network, loop_config.num_simulations)
        new_wins, old_wins, draws = pit(
            new_fn,
            old_fn,
            initial_state,
            num_games=loop_config.arena_games,
        )
        total = new_wins + old_wins + draws
        win_rate = new_wins / total if total > 0 else 0.0

        # ── Phase 4: 採用判定 ──────────────────────────────────────────
        adopted = win_rate >= loop_config.win_rate_threshold
        if adopted:
            best_network = new_network
            torch.save(best_network.state_dict(), model_path)

        progress_queue.put(
            {
                "type": "generation_done",
                "generation": generation + 1,
                "total": loop_config.num_generations,
                "policy_loss": round(losses["policy_loss"], 4),
                "value_loss": round(losses["value_loss"], 4),
                "total_loss": round(losses["total_loss"], 4),
                "new_wins": new_wins,
                "old_wins": old_wins,
                "draws": draws,
                "win_rate": round(win_rate, 3),
                "adopted": adopted,
                "data_size": len(data),
            }
        )

    progress_queue.put({"type": "done"})
