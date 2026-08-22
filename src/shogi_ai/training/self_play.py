"""Self-play data generation for AlphaZero-style training."""

from __future__ import annotations

import io
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import NetworkConfig
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


class _SerializedTrainingExample(NamedTuple):
    """A training example that does not use PyTorch shared-memory IPC."""

    state_bytes: bytes
    state_shape: tuple[int, ...]
    policy_bytes: bytes
    policy_shape: tuple[int, ...]
    value_target: float


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    num_games: int = 20
    num_simulations: int = 50
    temperature_threshold: int = 10  # この手数以降は温度を下げて最善手を選ぶ
    max_moves: int = 200
    batch_size: int = 8


class _SelfPlayTask(NamedTuple):
    """Pickleable arguments for a self-play worker process."""

    network_config: NetworkConfig
    state_dict_bytes: bytes
    initial_state: GameState
    config: SelfPlayConfig
    num_games: int
    device: str
    seed: int


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
    """
    examples: list[tuple[Tensor, Tensor, int]] = []
    mcts_config = MCTSConfig(num_simulations=config.num_simulations)
    mcts = MCTS(network, mcts_config)

    move_count = 0
    max_moves = config.max_moves  # 無限ループ防止（引き分けとして扱う）

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


def play_games_batched(
    network: DualHeadNetwork,
    initial_state: GameState,
    config: SelfPlayConfig,
    num_games: int,
) -> list[TrainingExample]:
    """Play several games together, batching MCTS neural evaluations."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    states = [initial_state for _ in range(num_games)]
    records: list[list[tuple[Tensor, Tensor, int]]] = [[] for _ in states]
    move_counts = [0] * num_games
    mcts = MCTS(network, MCTSConfig(num_simulations=config.num_simulations))

    while any(
        not state.is_terminal and count < config.max_moves
        for state, count in zip(states, move_counts)
    ):
        active = [
            index
            for index, (state, count) in enumerate(zip(states, move_counts))
            if not state.is_terminal and count < config.max_moves
        ]
        for batch_start in range(0, len(active), config.batch_size):
            batch_indices = active[batch_start : batch_start + config.batch_size]
            batch_states = [states[index] for index in batch_indices]
            temperature = (
                1.0
                if move_counts[batch_indices[0]] < config.temperature_threshold
                else 0.01
            )
            mcts.config = MCTSConfig(
                num_simulations=config.num_simulations,
                temperature=temperature,
            )
            batch_probs = mcts.search_batch(batch_states)
            for index, action_probs in zip(batch_indices, batch_probs):
                state = states[index]
                policy = torch.tensor(action_probs, dtype=torch.float32)
                records[index].append((state.to_tensor_planes(), policy, state.current_player))
                move = _select_move(action_probs, state.legal_moves())
                states[index] = state.apply_move(move)
                move_counts[index] += 1

    examples: list[TrainingExample] = []
    for state, game_records in zip(states, records):
        winner = state.winner
        for state_tensor, policy, player in game_records:
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            examples.append(TrainingExample(state_tensor, policy, value))
    return examples


def generate_training_data(
    network: DualHeadNetwork,
    initial_state: GameState,
    config: SelfPlayConfig,
    num_workers: int = 1,
    device_ids: list[int] | None = None,
) -> list[TrainingExample]:
    """Generate training data from multiple self-play games.

    複数の自己対局を行い、訓練データをまとめて返す。
    ``num_workers > 1`` では独立したゲームをプロセス並列化する。
    CUDA使用時は ``device_ids`` のGPUを各ワーカーへ1枚ずつ割り当てる。
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if device_ids is not None and len(device_ids) != num_workers:
        raise ValueError("device_ids length must match num_workers")
    if device_ids is not None:
        if not torch.cuda.is_available():
            raise ValueError("device_ids requires CUDA to be available")
        device_count = torch.cuda.device_count()
        if any(device_id < 0 or device_id >= device_count for device_id in device_ids):
            raise ValueError(f"device_ids must be between 0 and {device_count - 1}")

    if num_workers == 1:
        return [
            example
            for _ in range(config.num_games)
            for example in play_game(network, initial_state, config)
        ]

    devices = ["cpu"] * num_workers if device_ids is None else [
        f"cuda:{device_id}" for device_id in device_ids
    ]
    state_dict = {name: value.detach().cpu() for name, value in network.state_dict().items()}
    state_dict_buffer = io.BytesIO()
    torch.save(state_dict, state_dict_buffer)
    state_dict_bytes = state_dict_buffer.getvalue()
    tasks = [
        _SelfPlayTask(
            network.config,
            state_dict_bytes,
            initial_state,
            config,
            config.num_games // num_workers + (index < config.num_games % num_workers),
            device,
            random.randrange(2**63),
        )
        for index, device in enumerate(devices)
    ]
    tasks = [task for task in tasks if task.num_games > 0]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
        results = executor.map(_self_play_worker, tasks)
        return [
            _deserialize_training_example(example)
            for worker_examples in results
            for example in worker_examples
        ]


def _self_play_worker(task: _SelfPlayTask) -> list[_SerializedTrainingExample]:
    """Run a batch of self-play games in one CPU/GPU worker."""
    random.seed(task.seed)
    torch.set_num_threads(1)
    device = torch.device(task.device)
    network = DualHeadNetwork(task.network_config).to(device)
    state_dict = torch.load(
        io.BytesIO(task.state_dict_bytes),
        map_location=device,
        weights_only=True,
    )
    network.load_state_dict(state_dict)
    network.eval()

    examples = play_games_batched(network, task.initial_state, task.config, task.num_games)
    return [_serialize_training_example(example) for example in examples]


def _serialize_training_example(example: TrainingExample) -> _SerializedTrainingExample:
    """Serialize tensors as bytes to avoid PyTorch shared-memory handles."""
    state_tensor = example.state_tensor.detach().cpu().contiguous()
    policy_tensor = example.policy_target.detach().cpu().contiguous()
    return _SerializedTrainingExample(
        state_tensor.numpy().tobytes(),
        tuple(state_tensor.shape),
        policy_tensor.numpy().tobytes(),
        tuple(policy_tensor.shape),
        example.value_target,
    )


def _deserialize_training_example(
    example: _SerializedTrainingExample,
) -> TrainingExample:
    """Restore a serialized self-play example in the parent process."""
    state_tensor = torch.frombuffer(bytearray(example.state_bytes), dtype=torch.float32).reshape(
        example.state_shape
    )
    policy_tensor = torch.frombuffer(bytearray(example.policy_bytes), dtype=torch.float32).reshape(
        example.policy_shape
    )
    return TrainingExample(state_tensor, policy_tensor, example.value_target)


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
