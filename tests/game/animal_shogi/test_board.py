"""Tests for Board representation."""

from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.types import COLS, ROWS, PieceType, Player


class TestInitialPosition:
    def test_board_size(self) -> None:
        board = Board()
        assert len(board.squares) == ROWS * COLS

    def test_sente_lion_at_3_1(self) -> None:
        board = Board()
        piece = board.piece_at(3, 1)
        assert piece is not None
        assert piece.piece_type == PieceType.LION
        assert piece.owner == Player.SENTE

    def test_gote_lion_at_0_1(self) -> None:
        board = Board()
        piece = board.piece_at(0, 1)
        assert piece is not None
        assert piece.piece_type == PieceType.LION
        assert piece.owner == Player.GOTE

    def test_sente_pieces(self) -> None:
        board = Board()
        # Elephant at (3,0), Lion at (3,1), Giraffe at (3,2)
        assert board.piece_at(3, 0) == Piece(PieceType.ELEPHANT, Player.SENTE)
        assert board.piece_at(3, 2) == Piece(PieceType.GIRAFFE, Player.SENTE)
        # Chick at (2,1)
        assert board.piece_at(2, 1) == Piece(PieceType.CHICK, Player.SENTE)

    def test_gote_pieces(self) -> None:
        board = Board()
        # Giraffe at (0,0), Lion at (0,1), Elephant at (0,2)
        assert board.piece_at(0, 0) == Piece(PieceType.GIRAFFE, Player.GOTE)
        assert board.piece_at(0, 2) == Piece(PieceType.ELEPHANT, Player.GOTE)
        # Chick at (1,1)
        assert board.piece_at(1, 1) == Piece(PieceType.CHICK, Player.GOTE)

    def test_empty_squares(self) -> None:
        board = Board()
        for r, c in [(1, 0), (1, 2), (2, 0), (2, 2)]:
            assert board.piece_at(r, c) is None

    def test_empty_hands(self) -> None:
        board = Board()
        assert board.hands == ((), ())


class TestBoardOperations:
    def test_set_piece(self) -> None:
        board = Board()
        piece = Piece(PieceType.CHICK, Player.SENTE)
        new_board = board.set_piece(1, 0, piece)
        assert new_board.piece_at(1, 0) == piece
        assert board.piece_at(1, 0) is None  # Original unchanged

    def test_add_to_hand(self) -> None:
        board = Board()
        new_board = board.add_to_hand(Player.SENTE, PieceType.CHICK)
        assert new_board.hands[Player.SENTE.value] == (PieceType.CHICK,)
        assert board.hands[Player.SENTE.value] == ()  # Original unchanged

    def test_add_hen_to_hand_reverts_to_chick(self) -> None:
        board = Board()
        new_board = board.add_to_hand(Player.SENTE, PieceType.HEN)
        assert new_board.hands[Player.SENTE.value] == (PieceType.CHICK,)

    def test_remove_from_hand(self) -> None:
        board = Board(hands=((PieceType.CHICK, PieceType.GIRAFFE), ()))
        new_board = board.remove_from_hand(Player.SENTE, PieceType.CHICK)
        assert new_board.hands[Player.SENTE.value] == (PieceType.GIRAFFE,)

    def test_find_lion(self) -> None:
        board = Board()
        sente_lion = board.find_lion(Player.SENTE)
        gote_lion = board.find_lion(Player.GOTE)
        assert sente_lion == 3 * COLS + 1  # (3, 1) = index 10
        assert gote_lion == 0 * COLS + 1  # (0, 1) = index 1

    def test_find_lion_missing(self) -> None:
        """When lion is not on the board, return None."""
        squares = list(Board().squares)
        squares[1] = None  # Remove gote lion at (0, 1)
        board = Board(squares=tuple(squares))
        assert board.find_lion(Player.GOTE) is None
