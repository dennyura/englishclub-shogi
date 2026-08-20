"""Tests for the FastAPI web application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shogi_ai.web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestNewGame:
    def test_create_animal_game(self, client: TestClient) -> None:
        res = client.post("/api/new-game", json={"game_type": "animal"})
        assert res.status_code == 200
        data = res.json()
        assert "game_id" in data
        assert data["state"]["rows"] == 4
        assert data["state"]["cols"] == 3
        assert not data["state"]["is_terminal"]

    def test_create_full_game(self, client: TestClient) -> None:
        res = client.post("/api/new-game", json={"game_type": "full"})
        assert res.status_code == 200
        data = res.json()
        assert data["state"]["rows"] == 9
        assert data["state"]["cols"] == 9

    def test_invalid_game_type(self, client: TestClient) -> None:
        res = client.post("/api/new-game", json={"game_type": "chess"})
        assert res.status_code == 400


class TestMakeMove:
    def test_valid_move(self, client: TestClient) -> None:
        # Create game
        res = client.post(
            "/api/new-game",
            json={"game_type": "animal", "ai_type": "random"},
        )
        game_id = res.json()["game_id"]
        legal = res.json()["state"]["legal_moves"]

        # Make a move
        res = client.post(
            "/api/move",
            json={"game_id": game_id, "move": legal[0]},
        )
        assert res.status_code == 200
        data = res.json()
        assert "state" in data
        assert "ai_move" in data

    def test_invalid_move(self, client: TestClient) -> None:
        res = client.post(
            "/api/new-game",
            json={"game_type": "animal", "ai_type": "random"},
        )
        game_id = res.json()["game_id"]

        res = client.post(
            "/api/move",
            json={"game_id": game_id, "move": 999},
        )
        assert res.status_code == 400

    def test_game_not_found(self, client: TestClient) -> None:
        res = client.post(
            "/api/move",
            json={"game_id": "nonexistent", "move": 0},
        )
        assert res.status_code == 404


class TestGetState:
    def test_get_existing_game(self, client: TestClient) -> None:
        res = client.post("/api/new-game", json={"game_type": "animal"})
        game_id = res.json()["game_id"]

        res = client.get(f"/api/state/{game_id}")
        assert res.status_code == 200
        assert res.json()["rows"] == 4

    def test_get_nonexistent_game(self, client: TestClient) -> None:
        res = client.get("/api/state/nonexistent")
        assert res.status_code == 404


class TestIndex:
    def test_serves_html(self, client: TestClient) -> None:
        res = client.get("/")
        assert res.status_code == 200
        assert "将棋AI" in res.text


class TestGameFlow:
    def test_play_multiple_moves(self, client: TestClient) -> None:
        """Play several moves without errors."""
        res = client.post(
            "/api/new-game",
            json={"game_type": "animal", "ai_type": "random"},
        )
        game_id = res.json()["game_id"]

        for _ in range(5):
            state = res.json()["state"] if "state" in res.json() else res.json()
            if state.get("is_terminal", False):
                break
            legal = state.get("legal_moves", [])
            if not legal:
                break
            res = client.post(
                "/api/move",
                json={"game_id": game_id, "move": legal[0]},
            )
            assert res.status_code == 200
