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
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import NetworkConfig
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.arena import pit_parallel
from shogi_ai.training.self_play import SelfPlayConfig, generate_training_data
from shogi_ai.training.trainer import ReplayBuffer, Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainLoopConfig:
    """訓練ループの設定パラメータ。

    Attributes:
        num_generations:      世代数（1世代 = 自己対局+訓練+アリーナ）
        num_self_play_games:  1世代あたりの自己対局数
        num_simulations:      MCTSシミュレーション数（多いほど強いが遅い）
        arena_games:          アリーナ対戦数（新旧比較、偶数推奨）
        win_rate_threshold:   新モデル採用の勝率閾値（55%以上で採用）
        buffer_size:          リプレイバッファの最大局面数
        samples_per_generation: 1世代あたりの学習サンプル数
        batch_size:           学習時のミニバッチサイズ
        epochs_per_generation: 1世代あたりの学習エポック数
        max_training_hours:   学習時間の上限（時間、Noneで無制限）
        model_path:           最良モデルの保存先パス
        num_res_blocks:       残差ブロック数（Noneならnetwork_configの値を使用）
        self_play_batch_size: GPU推論へまとめて送る自己対局数
        arena_workers: アリーナ対戦ワーカープロセス数
        arena_device_ids: アリーナワーカーに割り当てるCUDAデバイス番号
        checkpoint_callback: 採用モデルを保存した直後に呼ぶコールバック
        log_callback: 全学習完了時に学習ログを渡すコールバック
    """

    num_generations: int = 10
    num_self_play_games: int = 5
    num_simulations: int = 25
    arena_games: int = 10
    win_rate_threshold: float = 0.55
    buffer_size: int = 30000
    samples_per_generation: int = 3000
    batch_size: int = 64
    epochs_per_generation: int = 10
    max_training_hours: float | None = None
    model_path: str = "best_model.pt"
    num_res_blocks: int | None = None
    self_play_workers: int = 1
    self_play_device_ids: tuple[int, ...] | None = None
    self_play_batch_size: int = 8
    arena_workers: int = 1
    arena_device_ids: tuple[int, ...] | None = None
    checkpoint_callback: Callable[[Path], None] | None = None
    log_callback: Callable[[str], None] | None = None


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


def _resolve_network_config(
    network_config: NetworkConfig,
    loop_config: TrainLoopConfig,
) -> NetworkConfig:
    """Apply loop-level network overrides while preserving other settings."""
    if loop_config.num_res_blocks is None:
        return network_config
    return replace(network_config, num_res_blocks=loop_config.num_res_blocks)


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
    if loop_config.max_training_hours is not None and loop_config.max_training_hours <= 0:
        raise ValueError("max_training_hours must be positive or None")
    if loop_config.num_res_blocks is not None and loop_config.num_res_blocks <= 0:
        raise ValueError("num_res_blocks must be positive or None")
    if loop_config.self_play_workers <= 0:
        raise ValueError("self_play_workers must be positive")
    if loop_config.self_play_batch_size <= 0:
        raise ValueError("self_play_batch_size must be positive")
    if loop_config.arena_workers <= 0:
        raise ValueError("arena_workers must be positive")
    if (
        loop_config.arena_device_ids is not None
        and len(loop_config.arena_device_ids) != loop_config.arena_workers
    ):
        raise ValueError("arena_device_ids length must match arena_workers")
    if (
        loop_config.self_play_device_ids is not None
        and len(loop_config.self_play_device_ids) != loop_config.self_play_workers
    ):
        raise ValueError("self_play_device_ids length must match self_play_workers")

    model_path = Path(loop_config.model_path)

    started_at = time.monotonic()

    def time_limit_reached() -> bool:
        if loop_config.max_training_hours is None:
            return False
        return time.monotonic() - started_at >= loop_config.max_training_hours * 3600

    device = _get_device()

    network_config = _resolve_network_config(network_config, loop_config)

    # 最良モデルを初期化（または保存済みモデルから続きを再開）
    best_network = DualHeadNetwork(network_config).to(device)
    if model_path.is_file():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        best_network.load_state_dict(state_dict)
    else:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_network.state_dict(), model_path)
        logger.info("Created initial model checkpoint: %s", model_path)

    trainer_config = TrainerConfig(
        buffer_size=loop_config.buffer_size,
        samples_per_generation=loop_config.samples_per_generation,
        batch_size=loop_config.batch_size,
        epochs_per_generation=loop_config.epochs_per_generation,
    )
    replay_buffer = ReplayBuffer(trainer_config.buffer_size)
    self_play_config = SelfPlayConfig(
        num_games=loop_config.num_self_play_games,
        num_simulations=loop_config.num_simulations,
        batch_size=loop_config.self_play_batch_size,
    )

    termination_reason = "generation_limit"
    log_lines: list[str] = []
    for generation in range(loop_config.num_generations):
        if stop_event.is_set():
            progress_queue.put({"type": "stopped"})
            return
        if time_limit_reached():
            termination_reason = "time_limit"
            break

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
        data = generate_training_data(
            best_network,
            initial_state,
            self_play_config,
            num_workers=loop_config.self_play_workers,
            device_ids=(
                list(loop_config.self_play_device_ids)
                if loop_config.self_play_device_ids is not None
                else None
            ),
        )
        replay_buffer.add(data)
        sampled_data = replay_buffer.sample(trainer_config.samples_per_generation)

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
                "data_size": len(sampled_data),
                "buffer_size": len(replay_buffer),
            }
        )

        new_network = copy.deepcopy(best_network).to(device)
        trainer = Trainer(new_network, trainer_config, device)
        losses = trainer.train(sampled_data)

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
        new_wins, old_wins, draws = pit_parallel(
            new_network,
            best_network,
            initial_state,
            num_games=loop_config.arena_games,
            num_simulations=loop_config.num_simulations,
            num_workers=loop_config.arena_workers,
            device_ids=(
                list(loop_config.arena_device_ids)
                if loop_config.arena_device_ids is not None
                else None
            ),
        )
        total = new_wins + old_wins + draws
        win_rate = new_wins / total if total > 0 else 0.0

        # ── Phase 4: 採用判定 ──────────────────────────────────────────
        adopted = win_rate >= loop_config.win_rate_threshold
        if adopted:
            best_network = new_network
            torch.save(best_network.state_dict(), model_path)
            if loop_config.checkpoint_callback is not None:
                loop_config.checkpoint_callback(model_path)

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
                "data_size": len(sampled_data),
                "buffer_size": len(replay_buffer),
            }
        )
        generation_log = (
            f"Generation {generation + 1}/{loop_config.num_generations} completed: "
            f"policy_loss={losses['policy_loss']:.4f}, "
            f"value_loss={losses['value_loss']:.4f}, "
            f"total_loss={losses['total_loss']:.4f}, "
            f"new_wins={new_wins}, old_wins={old_wins}, draws={draws}, "
            f"win_rate={win_rate:.3f}, adopted={adopted}, "
            f"data_size={len(sampled_data)}, buffer_size={len(replay_buffer)}"
        )
        log_lines.append(generation_log)
        logger.info(generation_log)
        print(
            f"Generation {generation + 1}/{loop_config.num_generations} completed: "
            f"loss={losses['total_loss']:.4f}, win_rate={win_rate:.3f}, "
            f"adopted={adopted}, data={len(sampled_data)}",
            flush=True,
        )

        if time_limit_reached():
            termination_reason = "time_limit"
            break

    log_lines.append(f"Training completed: reason={termination_reason}")
    if loop_config.log_callback is not None:
        loop_config.log_callback("\n".join(log_lines) + "\n")
    progress_queue.put({"type": "done", "reason": termination_reason})
