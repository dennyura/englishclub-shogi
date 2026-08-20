"""Tests for move generation and application."""

from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.moves import (
    apply_move,
    decode_move,
    encode_board_move,
    encode_drop_move,
    legal_moves,
)
from shogi_ai.game.animal_shogi.types import COLS, PieceType, Player


class TestMoveEncoding:
    def test_board_move_roundtrip(self) -> None:
        move = encode_board_move(10, 7)  # (3,1) -> (2,1)
        info = decode_move(move)
        assert info["type"] == "board"
        assert info["from"] == (3, 1)
        assert info["to"] == (2, 1)

    def test_drop_move_roundtrip(self) -> None:
        move = encode_drop_move(PieceType.CHICK, 3)  # Drop chick at (1,0)
        info = decode_move(move)
        assert info["type"] == "drop"
        assert info["piece_type"] == PieceType.CHICK
        assert info["to"] == (1, 0)


class TestLegalMoves:
    def test_initial_position_sente(self) -> None:
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        assert len(moves) > 0

    def test_initial_sente_chick_can_advance(self) -> None:
        """Sente's chick at (2,1) can move to (1,1) capturing gote's chick."""
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        chick_advance = encode_board_move(2 * COLS + 1, 1 * COLS + 1)
        assert chick_advance in moves

    def test_initial_sente_lion_moves(self) -> None:
        """Sente's lion at (3,1) can move to (2,0) and (2,2)."""
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        lion_idx = 3 * COLS + 1  # (3,1) = 10
        # Lion can go to (2,0)=6, (2,2)=8 (not (2,1) blocked by own chick,
        # not (3,0) own elephant, not (3,2) own giraffe)
        assert encode_board_move(lion_idx, 6) in moves
        assert encode_board_move(lion_idx, 8) in moves

    def test_cannot_capture_own_piece(self) -> None:
        """Sente's lion should not be able to move to squares with own pieces."""
        board = Board()
        moves = legal_moves(board, Player.SENTE)
        lion_idx = 10
        own_elephant = 9  # (3,0)
        own_giraffe = 11  # (3,2)
        own_chick = 7  # (2,1)
        assert encode_board_move(lion_idx, own_elephant) not in moves
        assert encode_board_move(lion_idx, own_giraffe) not in moves
        assert encode_board_move(lion_idx, own_chick) not in moves

    def test_drop_moves_available_with_hand(self) -> None:
        """If player has pieces in hand, drop moves should be generated."""
        board = Board(hands=((PieceType.CHICK,), ()))
        moves = legal_moves(board, Player.SENTE)
        # Count drop moves: should be able to drop on all empty squares
        drop_moves = [m for m in moves if m >= 144]
        # Initial board has 4 empty squares
        assert len(drop_moves) == 4

    def test_no_drop_on_occupied_square(self) -> None:
        """Cannot drop a piece on an occupied square."""
        board = Board(hands=((PieceType.CHICK,), ()))
        moves = legal_moves(board, Player.SENTE)
        # Square (0,1) has gote's lion, index 1
        drop_on_lion = encode_drop_move(PieceType.CHICK, 1)
        assert drop_on_lion not in moves


class TestApplyMove:
    def test_simple_move(self) -> None:
        """Move sente's chick from (2,1) to (1,1) capturing gote's chick."""
        board = Board()
        move = encode_board_move(2 * COLS + 1, 1 * COLS + 1)
        new_board = apply_move(board, Player.SENTE, move)

        # Old position empty
        assert new_board.piece_at(2, 1) is None
        # New position has sente's chick
        piece = new_board.piece_at(1, 1)
        assert piece is not None
        assert piece.piece_type == PieceType.CHICK
        assert piece.owner == Player.SENTE
        # Captured chick in hand
        assert PieceType.CHICK in new_board.hands[Player.SENTE.value]

    def test_promotion_chick_to_hen(self) -> None:
        """Sente's chick promotes to hen when reaching row 0."""
        # Place sente chick at (1, 0) with clear path to (0, 0)
        squares = list(Board().squares)
        squares[1 * COLS + 0] = Piece(PieceType.CHICK, Player.SENTE)
        # Clear (0, 0) for the test
        squares[0 * COLS + 0] = None
        board = Board(squares=tuple(squares))

        move = encode_board_move(1 * COLS + 0, 0 * COLS + 0)
        new_board = apply_move(board, Player.SENTE, move)

        piece = new_board.piece_at(0, 0)
        assert piece is not None
        assert piece.piece_type == PieceType.HEN
        assert piece.owner == Player.SENTE

    def test_gote_promotion(self) -> None:
        """Gote's chick promotes to hen when reaching row 3."""
        squares = list(Board().squares)
        squares[2 * COLS + 0] = Piece(PieceType.CHICK, Player.GOTE)
        squares[3 * COLS + 0] = None  # Clear elephant
        board = Board(squares=tuple(squares))

        move = encode_board_move(2 * COLS + 0, 3 * COLS + 0)
        new_board = apply_move(board, Player.GOTE, move)

        piece = new_board.piece_at(3, 0)
        assert piece is not None
        assert piece.piece_type == PieceType.HEN

    def test_drop_move(self) -> None:
        """Drop a chick from hand onto an empty square."""
        board = Board(hands=((PieceType.CHICK,), ()))
        # Drop on (1, 0) which is empty
        move = encode_drop_move(PieceType.CHICK, 1 * COLS + 0)
        new_board = apply_move(board, Player.SENTE, move)

        piece = new_board.piece_at(1, 0)
        assert piece is not None
        assert piece.piece_type == PieceType.CHICK
        assert piece.owner == Player.SENTE
        assert PieceType.CHICK not in new_board.hands[Player.SENTE.value]


class TestGoteMovement:
    def test_gote_chick_moves_down(self) -> None:
        """Gote's chick should move downward (positive row direction)."""
        board = Board()
        moves = legal_moves(board, Player.GOTE)
        # Gote's chick at (1,1) should be able to move to (2,1)
        chick_advance = encode_board_move(1 * COLS + 1, 2 * COLS + 1)
        assert chick_advance in moves


class TestMoveEncodingBoundary:
    """手エンコードの境界値テスト。"""

    def test_board_move_minimum_index(self) -> None:
        """最小インデックス (0, 0) → (0, 1) のエンコード/デコード。"""
        move = encode_board_move(0, 1)
        info = decode_move(move)
        assert info["type"] == "board"
        assert info["from"] == (0, 0)
        assert info["to"] == (0, 1)

    def test_board_move_maximum_index(self) -> None:
        """最大インデックス (11, 10) のエンコード/デコード（142が最大値）。"""
        move = encode_board_move(11, 10)
        assert move == 142  # 11*12 + 10 = 132 + 10 = 142
        info = decode_move(move)
        assert info["type"] == "board"

    def test_drop_move_boundary_first_type(self) -> None:
        """打ち駒の最小値（ひよこを盤面インデックス0に打つ）= 144。"""
        from shogi_ai.game.animal_shogi.moves import DROP_OFFSET

        move = encode_drop_move(PieceType.CHICK, 0)
        assert move == DROP_OFFSET  # = 144
        info = decode_move(move)
        assert info["type"] == "drop"
        assert info["piece_type"] == PieceType.CHICK
        assert info["to"] == (0, 0)

    def test_drop_move_maximum_value(self) -> None:
        """打ち駒の最大値（ぞうを盤面インデックス11に打つ）= 179。"""
        from shogi_ai.game.animal_shogi.moves import DROP_OFFSET

        move = encode_drop_move(PieceType.ELEPHANT, 11)
        assert move == DROP_OFFSET + 2 * 12 + 11  # = 144 + 24 + 11 = 179
        info = decode_move(move)
        assert info["type"] == "drop"
        assert info["piece_type"] == PieceType.ELEPHANT

    def test_drop_move_threshold_boundary(self) -> None:
        """DROP_OFFSET - 1 は盤上の手、DROP_OFFSET は打ち駒と判定される。"""
        from shogi_ai.game.animal_shogi.moves import DROP_OFFSET

        board_move_info = decode_move(DROP_OFFSET - 1)
        assert board_move_info["type"] == "board"

        drop_move_info = decode_move(DROP_OFFSET)
        assert drop_move_info["type"] == "drop"

    def test_all_drop_types_roundtrip(self) -> None:
        """打てる3駒種（ひよこ・きりん・ぞう）すべてがラウンドトリップできる。"""
        for pt in [PieceType.CHICK, PieceType.GIRAFFE, PieceType.ELEPHANT]:
            move = encode_drop_move(pt, 6)
            info = decode_move(move)
            assert info["piece_type"] == pt
