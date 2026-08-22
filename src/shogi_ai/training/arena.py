"""Arena for evaluating player strength through head-to-head matches.

アリーナ: 2つのプレイヤー関数を対戦させて強さを評価するモジュール。
AlphaZero では新旧ネットワークを対戦させ、勝率が閾値を超えれば新ネットワークに更新する。
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import torch

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import NetworkConfig
from shogi_ai.model.network import DualHeadNetwork


class _ArenaTask(NamedTuple):
    network_config: NetworkConfig
    player1_state: dict[str, torch.Tensor]
    player2_state: dict[str, torch.Tensor]
    initial_state: GameState
    num_games: int
    num_simulations: int
    max_moves: int
    device: str


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


def pit_parallel(
    player1: DualHeadNetwork,
    player2: DualHeadNetwork,
    initial_state: GameState,
    num_games: int = 50,
    num_simulations: int = 25,
    max_moves: int = 200,
    num_workers: int = 2,
    device_ids: list[int] | None = None,
) -> tuple[int, int, int]:
    """Evaluate two networks with independent arena worker processes."""
    if num_games <= 0 or num_workers <= 0:
        raise ValueError("num_games and num_workers must be positive")
    if device_ids is not None:
        if len(device_ids) != num_workers:
            raise ValueError("device_ids length must match num_workers")
        if not torch.cuda.is_available():
            raise ValueError("device_ids requires CUDA to be available")
        count = torch.cuda.device_count()
        if any(device_id < 0 or device_id >= count for device_id in device_ids):
            raise ValueError(f"device_ids must be between 0 and {count - 1}")

    devices = ["cpu"] * num_workers if device_ids is None else [
        f"cuda:{device_id}" for device_id in device_ids
    ]
    tasks = [
        _ArenaTask(
            player1.config,
            {name: value.detach().cpu() for name, value in player1.state_dict().items()},
            {name: value.detach().cpu() for name, value in player2.state_dict().items()},
            initial_state,
            num_games // num_workers + (index < num_games % num_workers),
            num_simulations,
            max_moves,
            device,
        )
        for index, device in enumerate(devices)
    ]
    tasks = [task for task in tasks if task.num_games > 0]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
        results = list(executor.map(_arena_worker, tasks))

    return tuple(sum(result[index] for result in results) for index in range(3))


def _arena_worker(task: _ArenaTask) -> tuple[int, int, int]:
    torch.set_num_threads(1)
    device = torch.device(task.device)
    player1 = DualHeadNetwork(task.network_config).to(device)
    player2 = DualHeadNetwork(task.network_config).to(device)
    player1.load_state_dict(task.player1_state)
    player2.load_state_dict(task.player2_state)
    player1.eval()
    player2.eval()
    player1_mcts = MCTS(player1, MCTSConfig(task.num_simulations, temperature=0.01))
    player2_mcts = MCTS(player2, MCTSConfig(task.num_simulations, temperature=0.01))

    def player1_fn(state: GameState) -> int:
        probs = player1_mcts.search(state)
        return max(state.legal_moves(), key=lambda move: probs[move])

    def player2_fn(state: GameState) -> int:
        probs = player2_mcts.search(state)
        return max(state.legal_moves(), key=lambda move: probs[move])

    return pit(player1_fn, player2_fn, task.initial_state, task.num_games, task.max_moves)
