"""Generate supervised training data from Minimax games."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from torch import Tensor

from shogi_ai.engine.minimax import minimax_scores
from shogi_ai.game.protocol import GameState
from shogi_ai.training.self_play import TrainingExample


def scores_to_policy(
    state: GameState,
    scored_moves: list[tuple[int, float]],
    top_k: int = 3,
    temperature: float = 100.0,
) -> Tensor:
    """Convert the top Minimax scores into an action probability tensor."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    policy = torch.zeros(state.action_space_size, dtype=torch.float32)
    candidates = scored_moves[:top_k]
    if not candidates:
        return policy

    best_score = candidates[0][1]
    weights = [math.exp((score - best_score) / temperature) for _, score in candidates]
    total = sum(weights)

    for (move, _), weight in zip(candidates, weights):
        policy[move] = weight / total

    return policy


def select_minimax_move(
    scored_moves: list[tuple[int, float]],
    move_count: int,
    opening_moves: int = 6,
    top_k: int = 3,
    temperature: float = 100.0,
) -> int:
    """Sample a top Minimax move during the opening, then play best move."""
    if opening_moves < 0:
        raise ValueError("opening_moves must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scored_moves:
        raise ValueError("scored_moves must not be empty")

    candidates = scored_moves[:top_k]
    if move_count >= opening_moves or len(candidates) == 1:
        return candidates[0][0]

    best_score = candidates[0][1]
    weights = [math.exp((score - best_score) / temperature) for _, score in candidates]
    return random.choices(
        [move for move, _ in candidates],
        weights=weights,
        k=1,
    )[0]


def play_minimax_game(
    initial_state: GameState,
    depth: int = 4,
    opening_moves: int = 6,
    top_k: int = 3,
    temperature: float = 100.0,
    max_moves: int = 200,
) -> list[TrainingExample]:
    """Generate supervised examples from one Minimax-guided game."""
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")

    state = initial_state
    records: list[tuple[Tensor, Tensor, int]] = []
    move_count = 0

    while not state.is_terminal and move_count < max_moves:
        scored_moves = minimax_scores(state, depth)
        policy = scores_to_policy(state, scored_moves, top_k, temperature)
        records.append((state.to_tensor_planes(), policy, state.current_player))

        move = select_minimax_move(
            scored_moves,
            move_count,
            opening_moves,
            top_k,
            temperature,
        )
        state = state.apply_move(move)
        move_count += 1

    winner = state.winner
    examples: list[TrainingExample] = []
    for state_tensor, policy, player in records:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        examples.append(TrainingExample(state_tensor, policy, value))

    return examples


def generate_minimax_data(
    initial_state: GameState,
    num_games: int,
    depth: int = 4,
    opening_moves: int = 6,
    top_k: int = 3,
    temperature: float = 100.0,
    max_moves: int = 512,
) -> list[TrainingExample]:
    """Generate examples from multiple Minimax-guided games."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")

    examples: list[TrainingExample] = []
    for _ in range(num_games):
        examples.extend(
            play_minimax_game(
                initial_state,
                depth,
                opening_moves,
                top_k,
                temperature,
                max_moves,
            )
        )
    return examples


def save_training_examples(
    examples: list[TrainingExample],
    output_path: str | Path,
) -> None:
    """Save examples as tensors that can be loaded by the training job."""
    if not examples:
        raise ValueError("examples must not be empty")

    torch.save(
        {
            "state_tensors": torch.stack([example.state_tensor for example in examples]),
            "policy_targets": torch.stack([example.policy_target for example in examples]),
            "value_targets": torch.tensor(
                [example.value_target for example in examples],
                dtype=torch.float32,
            ),
        },
        Path(output_path),
    )
