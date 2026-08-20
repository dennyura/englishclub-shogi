"""Tests for self-play data generation."""

from __future__ import annotations

from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.self_play import (
    SelfPlayConfig,
    TrainingExample,
    generate_training_data,
    play_game,
)


def _make_network() -> DualHeadNetwork:
    net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
    net.eval()
    return net


class TestPlayGame:
    def test_returns_training_examples(self) -> None:
        net = _make_network()
        config = SelfPlayConfig(num_games=1, num_simulations=5)
        state = AnimalShogiState()
        examples = play_game(net, state, config)

        assert len(examples) > 0
        for ex in examples:
            assert isinstance(ex, TrainingExample)

    def test_example_shapes(self) -> None:
        net = _make_network()
        config = SelfPlayConfig(num_games=1, num_simulations=5)
        state = AnimalShogiState()
        examples = play_game(net, state, config)

        for ex in examples:
            assert ex.state_tensor.shape == (14, 4, 3)
            assert ex.policy_target.shape == (180,)
            assert -1.0 <= ex.value_target <= 1.0

    def test_policy_sums_to_one(self) -> None:
        net = _make_network()
        config = SelfPlayConfig(num_games=1, num_simulations=5)
        state = AnimalShogiState()
        examples = play_game(net, state, config)

        for ex in examples:
            total = ex.policy_target.sum().item()
            assert abs(total - 1.0) < 0.02, f"Policy sum {total} != 1.0"

    def test_value_targets_assigned(self) -> None:
        net = _make_network()
        config = SelfPlayConfig(num_games=1, num_simulations=5)
        state = AnimalShogiState()
        examples = play_game(net, state, config)

        # At least some examples should have non-zero value
        values = [ex.value_target for ex in examples]
        assert any(v != 0.0 for v in values) or len(values) == 0


class TestGenerateTrainingData:
    def test_multiple_games(self) -> None:
        net = _make_network()
        config = SelfPlayConfig(num_games=3, num_simulations=5)
        state = AnimalShogiState()
        examples = generate_training_data(net, state, config)

        # 3 games, each game has multiple positions
        assert len(examples) >= 3
