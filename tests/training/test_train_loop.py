"""Tests for the AlphaZero training loop."""

from __future__ import annotations

import queue
import threading

from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.training.train_loop import (
    TrainLoopConfig,
    _resolve_network_config,
    run_training,
)


def test_missing_model_is_created_and_training_can_start(tmp_path) -> None:
    """Training should create an initial checkpoint when the model file is missing."""
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    missing_path = tmp_path / "missing.pt"
    config = TrainLoopConfig(model_path=str(missing_path), num_generations=0)

    run_training(
        AnimalShogiState(),
        ANIMAL_SHOGI_CONFIG,
        config,
        progress_queue,
        threading.Event(),
    )

    message = progress_queue.get_nowait()
    assert message["type"] == "done"
    assert message["reason"] == "generation_limit"
    assert missing_path.is_file()


def test_num_res_blocks_can_be_overridden_by_loop_config() -> None:
    config = TrainLoopConfig(num_res_blocks=8)
    network_config = _resolve_network_config(ANIMAL_SHOGI_CONFIG, config)

    assert network_config.num_res_blocks == 8


def test_num_res_blocks_must_be_positive(tmp_path) -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    config = TrainLoopConfig(model_path=str(tmp_path / "missing.pt"), num_res_blocks=0)

    try:
        run_training(
            AnimalShogiState(),
            ANIMAL_SHOGI_CONFIG,
            config,
            progress_queue,
            threading.Event(),
        )
    except ValueError as error:
        assert str(error) == "num_res_blocks must be positive or None"
    else:
        raise AssertionError("run_training should reject non-positive num_res_blocks")
