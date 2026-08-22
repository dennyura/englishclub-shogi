"""Tests for process-parallel arena evaluation."""

from __future__ import annotations

from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.arena import pit_parallel


def test_pit_parallel_returns_all_game_results() -> None:
    player1 = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
    player2 = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)

    result = pit_parallel(
        player1,
        player2,
        AnimalShogiState(),
        num_games=2,
        num_simulations=1,
        max_moves=1,
        num_workers=2,
    )

    assert sum(result) == 2
