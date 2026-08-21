"""Tests for the AlphaZero training loop."""

from __future__ import annotations

import logging
import queue
import threading

from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.training.train_loop import TrainLoopConfig, run_training


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
