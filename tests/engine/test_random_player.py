"""Tests for random player."""

from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState


def test_returns_legal_move() -> None:
    state = AnimalShogiState()
    move = random_move(state)
    assert move in state.legal_moves()


def test_game_completes() -> None:
    """Random vs random game should terminate within 200 moves."""
    state = AnimalShogiState()
    for _ in range(200):
        if state.is_terminal:
            break
        move = random_move(state)
        state = state.apply_move(move)
    assert state.is_terminal
