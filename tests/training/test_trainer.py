"""Tests for the training loop and arena."""

from __future__ import annotations

import pytest
import torch

from shogi_ai.engine.minimax import minimax_move
from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.arena import pit
from shogi_ai.training.self_play import (
    SelfPlayConfig,
    TrainingExample,
    generate_training_data,
)
from shogi_ai.training.trainer import Trainer, TrainerConfig


def _make_network() -> DualHeadNetwork:
    return DualHeadNetwork(ANIMAL_SHOGI_CONFIG)


class TestTrainer:
    def test_loss_decreases_with_training(self) -> None:
        """Training should reduce loss over epochs."""
        net = _make_network()
        device = torch.device("cpu")
        trainer = Trainer(
            net,
            TrainerConfig(epochs_per_generation=5, batch_size=8),
            device,
        )

        # Create simple training data
        examples = []
        state = AnimalShogiState()
        for _ in range(20):
            tensor = state.to_tensor_planes()
            policy = torch.zeros(180)
            legal = state.legal_moves()
            if legal:
                policy[legal[0]] = 1.0
            examples.append(TrainingExample(tensor, policy, 1.0))

        losses1 = trainer.train(examples)
        losses2 = trainer.train(examples)

        # Second round should have lower loss (network memorizes)
        assert losses2["total_loss"] < losses1["total_loss"]

    def test_empty_examples(self) -> None:
        net = _make_network()
        device = torch.device("cpu")
        trainer = Trainer(net, TrainerConfig(), device)
        losses = trainer.train([])
        assert losses["total_loss"] == 0.0


class TestArena:
    def test_random_vs_random(self) -> None:
        """Two random players should have roughly even results."""
        state = AnimalShogiState()
        wins, losses, draws = pit(random_move, random_move, state, num_games=20)
        assert wins + losses + draws == 20

    def test_minimax_vs_random(self) -> None:
        """Minimax should dominate random."""
        state = AnimalShogiState()

        def minimax_fn(s: AnimalShogiState) -> int:
            return minimax_move(s, depth=2)

        wins, losses, draws = pit(minimax_fn, random_move, state, num_games=10)
        assert wins > losses  # Minimax should win more

    def test_game_count_correct(self) -> None:
        state = AnimalShogiState()
        wins, losses, draws = pit(random_move, random_move, state, num_games=10)
        assert wins + losses + draws == 10


class TestIntegration:
    @pytest.mark.slow
    def test_self_play_train_cycle(self) -> None:
        """Full cycle: self-play → train → verify loss decreases."""
        net = _make_network()
        device = torch.device("cpu")
        state = AnimalShogiState()

        # Generate data
        config = SelfPlayConfig(num_games=3, num_simulations=10)
        examples = generate_training_data(net, state, config)
        assert len(examples) > 0

        # Train
        trainer = Trainer(
            net,
            TrainerConfig(epochs_per_generation=3, batch_size=16),
            device,
        )
        losses = trainer.train(examples)
        assert losses["total_loss"] > 0
