"""Tests for AnimalShogiState."""

from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.moves import encode_board_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import COLS, ROWS, PieceType, Player
from shogi_ai.game.protocol import GameState


class TestProtocolCompliance:
    def test_implements_game_state(self) -> None:
        state = AnimalShogiState()
        assert isinstance(state, GameState)


class TestInitialState:
    def test_sente_starts(self) -> None:
        state = AnimalShogiState()
        assert state.current_player == Player.SENTE.value

    def test_not_terminal(self) -> None:
        state = AnimalShogiState()
        assert not state.is_terminal

    def test_winner_none(self) -> None:
        state = AnimalShogiState()
        assert state.winner is None

    def test_has_legal_moves(self) -> None:
        state = AnimalShogiState()
        assert len(state.legal_moves()) > 0


class TestApplyMove:
    def test_player_alternates(self) -> None:
        state = AnimalShogiState()
        moves = state.legal_moves()
        new_state = state.apply_move(moves[0])
        assert new_state.current_player == Player.GOTE.value

    def test_apply_does_not_mutate(self) -> None:
        state = AnimalShogiState()
        moves = state.legal_moves()
        state.apply_move(moves[0])
        # Original should be unchanged
        assert state.current_player == Player.SENTE.value
        assert state.board == Board()


class TestTerminalConditions:
    def test_lion_capture_wins(self) -> None:
        """Capturing the opponent's lion should end the game."""
        # Set up a board where sente can capture gote's lion
        squares = [None] * (ROWS * COLS)
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)  # (0,1)
        squares[1 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)  # (1,1)
        board = Board(squares=tuple(squares))
        state = AnimalShogiState(board=board)

        # Sente captures gote's lion
        move = encode_board_move(1 * COLS + 1, 0 * COLS + 1)
        new_state = state.apply_move(move)

        assert new_state.is_terminal
        assert new_state.winner == Player.SENTE.value

    def test_try_rule_wins(self) -> None:
        """Moving lion to opponent's back rank without being capturable wins."""
        # Sente's lion at row 1, gote's lion far away
        squares = [None] * (ROWS * COLS)
        squares[1 * COLS + 0] = Piece(PieceType.LION, Player.SENTE)  # (1,0)
        squares[3 * COLS + 2] = Piece(PieceType.LION, Player.GOTE)  # (3,2)
        board = Board(squares=tuple(squares))
        state = AnimalShogiState(board=board)

        # Sente moves lion to (0,0) — gote's back rank
        move = encode_board_move(1 * COLS + 0, 0 * COLS + 0)
        new_state = state.apply_move(move)

        assert new_state.is_terminal
        assert new_state.winner == Player.SENTE.value

    def test_try_rule_fails_if_capturable(self) -> None:
        """Try rule doesn't win if the lion can be captured."""
        squares = [None] * (ROWS * COLS)
        squares[1 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)  # (1,1)
        # Gote's lion can reach (0,0)
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)  # (0,1)
        board = Board(squares=tuple(squares))
        state = AnimalShogiState(board=board)

        # Sente moves lion to (0,0) — but gote can capture at (0,0)
        move = encode_board_move(1 * COLS + 1, 0 * COLS + 0)
        new_state = state.apply_move(move)

        # Not a try win because gote can capture
        assert new_state.winner != Player.SENTE.value


class TestTensorPlanes:
    def test_shape(self) -> None:
        state = AnimalShogiState()
        tensor = state.to_tensor_planes()
        assert tensor.shape == (14, ROWS, COLS)

    def test_sente_lion_plane(self) -> None:
        state = AnimalShogiState()
        tensor = state.to_tensor_planes()
        # Lion is PieceType 3, sente's lion at (3,1)
        assert tensor[PieceType.LION.value, 3, 1] == 1.0

    def test_turn_indicator(self) -> None:
        state = AnimalShogiState()
        tensor = state.to_tensor_planes()
        # SENTE's turn: plane 13 should be all 1s
        assert tensor[13].sum() == ROWS * COLS
