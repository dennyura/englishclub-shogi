"""Tests for the AlphaZero training loop."""

from __future__ import annotations

import logging
import queue
import threading

from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.training.train_loop import (
    TrainLoopConfig,
    _resolve_network_config,
    run_training,
)


def test_missing_model_stops_training_and_logs_reason(tmp_path, caplog) -> None:
    """Training should stop before self-play when the model file is missing."""
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    missing_path = tmp_path / "missing.pt"
    config = TrainLoopConfig(model_path=str(missing_path))

    with caplog.at_level(logging.ERROR):
        run_training(
            AnimalShogiState(),
            ANIMAL_SHOGI_CONFIG,
            config,
            progress_queue,
            threading.Event(),
        )

    message = progress_queue.get_nowait()
    assert message["type"] == "done"
    assert message["reason"] == "model_not_found"
    assert str(missing_path) in caplog.text


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
