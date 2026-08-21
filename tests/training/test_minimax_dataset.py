"""Tests for Minimax-supervised dataset generation."""

from __future__ import annotations

import random

import pytest
import torch

from shogi_ai.engine.minimax import minimax_scores
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.training.minimax_dataset import (
    generate_minimax_data,
    play_minimax_game,
    save_training_examples,
    scores_to_policy,
    select_minimax_move,
)
from shogi_ai.training.self_play import TrainingExample


def test_minimax_scores_are_sorted_and_legal() -> None:
    state = AnimalShogiState()
    scored_moves = minimax_scores(state, depth=1)

    assert {move for move, _ in scored_moves} == set(state.legal_moves())
    assert all(
        scored_moves[index][1] >= scored_moves[index + 1][1]
        for index in range(len(scored_moves) - 1)
    )


def test_scores_to_policy_masks_non_top_moves() -> None:
    state = AnimalShogiState()
    scored_moves = minimax_scores(state, depth=1)
    policy = scores_to_policy(state, scored_moves, top_k=2, temperature=100.0)

    assert policy.shape == (state.action_space_size,)
    assert torch.isclose(policy.sum(), torch.tensor(1.0))
    assert int((policy > 0).sum()) == 2
    assert all(policy[move] == 0 for move, _ in scored_moves[2:])


def test_play_minimax_game_returns_training_examples() -> None:
    examples = play_minimax_game(
        AnimalShogiState(),
        depth=1,
        opening_moves=2,
        top_k=2,
        temperature=100.0,
    )

    assert examples
    assert all(isinstance(example, TrainingExample) for example in examples)
    assert all(example.state_tensor.shape == (14, 4, 3) for example in examples)
    assert all(example.policy_target.shape == (180,) for example in examples)
    assert all(
        torch.isclose(example.policy_target.sum(), torch.tensor(1.0))
        for example in examples
    )
    assert all(-1.0 <= example.value_target <= 1.0 for example in examples)


def test_play_minimax_game_supports_full_shogi() -> None:
    examples = play_minimax_game(
        FullShogiState(),
        depth=1,
        opening_moves=1,
        top_k=2,
        temperature=100.0,
        max_moves=1,
    )

    assert len(examples) == 1
    assert examples[0].state_tensor.shape == (43, 9, 9)
    assert examples[0].policy_target.shape == (13689,)
    assert torch.isclose(examples[0].policy_target.sum(), torch.tensor(1.0))


def test_opening_sampling_can_select_a_non_best_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "choices", lambda population, weights, k: [population[-1]])
    move = select_minimax_move(
        [(10, 100.0), (20, 90.0)],
        move_count=0,
        opening_moves=1,
        top_k=2,
        temperature=100.0,
    )

    assert move == 20


def test_save_training_examples(tmp_path) -> None:
    examples = generate_minimax_data(AnimalShogiState(), num_games=1, depth=1)
    output_path = tmp_path / "dataset.pt"
    save_training_examples(examples, output_path)

    data = torch.load(output_path, weights_only=True)
    assert data["state_tensors"].shape[1:] == (14, 4, 3)
    assert data["policy_targets"].shape[1:] == (180,)
    assert data["value_targets"].shape == (len(examples),)
