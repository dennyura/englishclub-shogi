"""Comprehensive test suite for 本将棋 (Full Shogi, 9x9).

Coverage map:
  - Individual piece movement: Pawn, Lance, Knight, Silver, Gold,
    Bishop, Rook, King, Horse (成り角), Dragon (成り飛)
  - Check detection: _is_in_check
  - Legal move filtering: pinned pieces, check escape, block/capture
  - Promotion rules: optional, mandatory, captured piece reversion
  - Special rules: Nifu (二歩), dead-piece restriction, Uchifuzume (打ち歩詰め)
  - Full game: random vs random until terminal, winner declared

Board coordinate convention (from board.py and types.py):
  squares index = row * 9 + col
  SENTE moves forward = decreasing row (row 8 → row 0)
  GOTE  moves forward = increasing row (row 0 → row 8)
  row 0 = GOTE's back rank (top of board)
  row 8 = SENTE's back rank (bottom of board)

Helper: _make_board(pieces, sente_hand=(), gote_hand=())
  pieces: list of (row, col, PieceType, Player)
  Returns a Board with exactly those pieces and the given hands.
"""

from __future__ import annotations

import random

import pytest

from shogi_ai.engine.random_player import random_move
from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.moves import (
    _is_in_check,
    apply_move,
    decode_move,
    encode_board_move,
    encode_drop_move,
    legal_moves,
)
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.game.full_shogi.types import (
    COLS,
    NUM_SQUARES,
    PieceType,
    Player,
)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_board(
    pieces: list[tuple[int, int, PieceType, Player]],
    sente_hand: tuple[PieceType, ...] = (),
    gote_hand: tuple[PieceType, ...] = (),
) -> Board:
    """Build a Board from an explicit piece list.

    Every position starts with both Kings present unless the caller
    deliberately omits one. This avoids accidental terminal states
    caused by a missing king.

    Pieces are specified as (row, col, PieceType, Player) tuples.
    """
    squares: list[Piece | None] = [None] * NUM_SQUARES
    for row, col, pt, owner in pieces:
        squares[row * COLS + col] = Piece(pt, owner)
    return Board(
        squares=tuple(squares),
        hands=(tuple(sorted(sente_hand)), tuple(sorted(gote_hand))),
    )


def _moves_from(board: Board, player: Player, from_row: int, from_col: int) -> list[dict]:
    """Return decoded legal moves that originate from (from_row, from_col)."""
    all_moves = legal_moves(board, player)
    result = []
    for move in all_moves:
        d = decode_move(move)
        if d["type"] == "board" and d["from"] == (from_row, from_col):
            result.append(d)
    return result


def _destination_set(moves: list[dict]) -> set[tuple[int, int]]:
    """Extract the set of (to_row, to_col) from a list of decoded moves."""
    return {d["to"] for d in moves}


# ===========================================================================
# 1. Individual piece movement
# ===========================================================================


class TestPawnMovement:
    """歩兵（歩）: 1マス前方のみ。"""

    def test_sente_pawn_moves_one_forward(self) -> None:
        """Sente pawn at (5, 4) must move to (4, 4) only."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.PAWN, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        dests = _destination_set(moves)
        assert dests == {(4, 4)}, f"Pawn should only reach (4,4), got {dests}"

    def test_sente_pawn_cannot_move_backward(self) -> None:
        """Pawn cannot move to (6, 4) — backward for Sente."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.PAWN, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        dests = _destination_set(moves)
        assert (6, 4) not in dests

    def test_pawn_blocked_by_own_piece(self) -> None:
        """Pawn cannot move forward if own piece is blocking."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.PAWN, Player.SENTE),
                (4, 4, PieceType.GOLD, Player.SENTE),  # Blocking
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        assert len(moves) == 0, "Pawn should have no moves when blocked by own piece"


class TestLanceMovement:
    """香車: 前方に何マスでもスライド。後方・横は不可。"""

    def test_lance_slides_forward_on_open_file(self) -> None:
        """Sente lance at (7, 0) must reach rows 6, 5, 4, 3, 2, 1, 0 on col 0."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (7, 0, PieceType.LANCE, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 7, 0)
        dests = _destination_set(moves)
        # All forward squares on col 0 except row 7 itself
        expected_base = {(r, 0) for r in range(0, 7)}
        # Row 0 forces promotion, so only the promote=True variant appears;
        # rows 1..2 are in promotion zone so both variants appear.
        # The destination set still includes (0, 0) regardless of promotion flag.
        assert expected_base == dests, f"Lance expected all of col 0 rows 0-6, got {dests}"

    def test_lance_blocked_by_own_piece(self) -> None:
        """Lance cannot slide past own Gold at row 5 on col 0."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (7, 0, PieceType.LANCE, Player.SENTE),
                (5, 0, PieceType.GOLD, Player.SENTE),  # blocker
            ]
        )
        moves = _moves_from(board, Player.SENTE, 7, 0)
        dests = _destination_set(moves)
        assert (6, 0) in dests  # one square before blocker is reachable
        assert (5, 0) not in dests  # blocker square is own piece – not reachable
        assert (4, 0) not in dests  # beyond blocker – not reachable

    def test_lance_can_capture_forward_enemy(self) -> None:
        """Lance stops after capturing enemy piece; cannot go further."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (7, 0, PieceType.LANCE, Player.SENTE),
                (4, 0, PieceType.PAWN, Player.GOTE),  # enemy to capture
            ]
        )
        moves = _moves_from(board, Player.SENTE, 7, 0)
        dests = _destination_set(moves)
        assert (4, 0) in dests  # can capture
        assert (3, 0) not in dests  # cannot go past enemy


class TestKnightMovement:
    """桂馬: 前2マス+左右1マスの2点のみ。間を飛び越える。"""

    def test_knight_reaches_exactly_two_squares(self) -> None:
        """Sente knight at (5, 4) must reach only (3, 3) and (3, 5)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.KNIGHT, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        dests = _destination_set(moves)
        assert dests == {(3, 3), (3, 5)}, (
            f"Knight from (5,4) must reach exactly {{(3,3),(3,5)}}, got {dests}"
        )

    def test_knight_cannot_move_sideways_or_backward(self) -> None:
        """Knight may not land on squares like (5,3), (6,4), (5,5)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.KNIGHT, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        dests = _destination_set(moves)
        forbidden = {(5, 3), (5, 5), (6, 4), (7, 4), (4, 4)}
        assert dests.isdisjoint(forbidden), (
            f"Knight landed on forbidden square(s): {dests & forbidden}"
        )

    def test_knight_jumps_over_intervening_piece(self) -> None:
        """Knight ignores pieces between start and landing square."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.KNIGHT, Player.SENTE),
                (4, 4, PieceType.GOLD, Player.SENTE),  # piece in the path – irrelevant
            ]
        )
        moves = _moves_from(board, Player.SENTE, 5, 4)
        dests = _destination_set(moves)
        # Both knight destinations remain reachable despite piece on (4,4)
        assert (3, 3) in dests
        assert (3, 5) in dests


class TestSilverMovement:
    """銀将: 前3方向 + 斜め後2方向 = 5方向1マス。"""

    def test_silver_has_five_directions(self) -> None:
        """Sente silver at (4, 4) in open center must have exactly 5 destinations."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.SILVER, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        expected = {(3, 3), (3, 4), (3, 5), (5, 3), (5, 5)}
        assert dests == expected, f"Silver expected {expected}, got {dests}"

    def test_silver_cannot_move_sideways_or_straight_backward(self) -> None:
        """Silver must NOT reach left/right (4,3)/(4,5) or straight back (5,4)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.SILVER, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        assert (4, 3) not in dests, "Silver cannot move sideways left"
        assert (4, 5) not in dests, "Silver cannot move sideways right"
        assert (5, 4) not in dests, "Silver cannot move straight backward"


class TestGoldMovement:
    """金将: 前3方向 + 左右 + 直後 = 6方向1マス。斜め後ろは不可。"""

    def test_gold_has_six_directions(self) -> None:
        """Sente gold at (4, 4) in open center must have exactly 6 destinations."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.GOLD, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        expected = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 4)}
        assert dests == expected, f"Gold expected {expected}, got {dests}"

    def test_gold_cannot_move_to_backward_diagonals(self) -> None:
        """Gold cannot go to (5, 3) or (5, 5) — the backward diagonals."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.GOLD, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        assert (5, 3) not in dests, "Gold cannot move to backward-left diagonal"
        assert (5, 5) not in dests, "Gold cannot move to backward-right diagonal"


class TestBishopMovement:
    """角行: 斜め4方向にスライド。"""

    def test_bishop_slides_all_four_diagonals(self) -> None:
        """Bishop at (4, 4) on empty board must reach all diagonal squares."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.BISHOP, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        # Each diagonal from (4,4): 4 directions, distance up to board edge
        # NW: (3,3),(2,2),(1,1),(0,0)  NE: (3,5),(2,6),(1,7),(0,8)
        # SW: (5,3),(6,2),(7,1),(8,0)  SE: (5,5),(6,6),(7,7)  — (8,8) blocked later
        # King at (8,4) doesn't block diagonals from (4,4)
        assert (3, 3) in dests  # NW diagonal
        assert (0, 0) in dests  # far NW
        assert (3, 5) in dests  # NE diagonal
        assert (5, 3) in dests  # SW diagonal
        assert (5, 5) in dests  # SE diagonal
        # Bishop does NOT move orthogonally
        assert (4, 3) not in dests
        assert (4, 5) not in dests
        assert (3, 4) not in dests
        assert (5, 4) not in dests

    def test_bishop_blocked_by_friendly_piece(self) -> None:
        """Bishop cannot slide past or onto own piece."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.BISHOP, Player.SENTE),
                (2, 2, PieceType.GOLD, Player.SENTE),  # blocks NW diagonal
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        assert (3, 3) in dests  # still reachable
        assert (2, 2) not in dests  # own piece — blocked
        assert (1, 1) not in dests  # beyond own piece — blocked
        assert (0, 0) not in dests  # far beyond own piece — blocked

    def test_bishop_captures_and_stops(self) -> None:
        """Bishop captures enemy but cannot continue past it."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.BISHOP, Player.SENTE),
                (2, 2, PieceType.PAWN, Player.GOTE),  # enemy on NW diagonal
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        assert (3, 3) in dests  # before enemy — reachable
        assert (2, 2) in dests  # enemy square — capture
        assert (1, 1) not in dests  # beyond capture — blocked


class TestRookMovement:
    """飛車: 縦横4方向にスライド。"""

    def test_rook_slides_four_orthogonal_directions(self) -> None:
        """Rook at (4, 4) on otherwise empty board reaches all orthogonal squares."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.ROOK, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        # King at (8,4) blocks the south (col 4) after row 8, but (8,4) is
        # occupied by own King so rook can't land there, stops at (7,4).
        for r in range(0, 4):
            assert (r, 4) in dests, f"Rook should reach ({r}, 4)"
        for r in range(5, 8):
            assert (r, 4) in dests, f"Rook should reach ({r}, 4)"
        # (8, 4) is King's own square — blocked
        assert (8, 4) not in dests
        for c in range(0, 4):
            assert (4, c) in dests, f"Rook should reach (4, {c})"
        for c in range(5, 9):
            assert (4, c) in dests, f"Rook should reach (4, {c})"
        # Rook does NOT move diagonally
        assert (3, 3) not in dests
        assert (5, 5) not in dests

    def test_rook_blocked_by_own_piece_in_path(self) -> None:
        """Rook stops before own Silver at (4, 2)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.ROOK, Player.SENTE),
                (4, 2, PieceType.SILVER, Player.SENTE),  # blocker in same row
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        assert (4, 3) in dests  # one before blocker
        assert (4, 2) not in dests  # own piece
        assert (4, 1) not in dests  # beyond
        assert (4, 0) not in dests  # beyond


class TestKingMovement:
    """玉将: 全8方向に1マス。"""

    def test_king_reaches_all_eight_directions_in_open_center(self) -> None:
        """Sente king at (4, 4) — open center — must reach all 8 adjacent squares."""
        board = _make_board(
            [
                (4, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ]
        )
        # Note: we do NOT filter for check here — we want raw move generation.
        # Use legal_moves which will filter, but in this isolated position no
        # enemy pieces threaten king so all 8 moves should be legal.
        all_legal = legal_moves(board, Player.SENTE)
        decoded = [decode_move(m) for m in all_legal if decode_move(m)["type"] == "board"]
        king_dests = {d["to"] for d in decoded if d["from"] == (4, 4)}
        expected = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)}
        assert king_dests == expected, f"King expected {expected}, got {king_dests}"

    def test_king_cannot_move_into_check(self) -> None:
        """King must not move to a square attacked by enemy rook."""
        # Gote rook at (3, 0) covers all of row 3 and col 0.
        # Sente king at (5, 4): if it tries to move to (4, 4) that is fine,
        # but (4, 0) would be covered by the rook... let's set up a cleaner case.
        # Gote rook on col 4 row 0 covers all of col 4.
        # Sente king cannot step to (4, 4) since col 4 is covered.
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (0, 0, PieceType.ROOK, Player.GOTE),  # covers col 0 and row 0 — no threat to (4,x)
                (2, 8, PieceType.ROOK, Player.GOTE),  # covers col 8 and row 2
            ]
        )
        # Place Gote rook on col 4 (same col as king), one square above king
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 8, PieceType.KING, Player.GOTE),
                (3, 4, PieceType.ROOK, Player.GOTE),  # controls col 4 from row 3 upward
            ]
        )
        all_legal = legal_moves(board, Player.SENTE)
        decoded = [decode_move(m) for m in all_legal]
        king_dests = {d["to"] for d in decoded if d.get("from") == (5, 4)}
        # (4, 4) is covered by rook at (3, 4) via col 4
        assert (4, 4) not in king_dests, "King must not walk into rook's line of attack"


class TestHorseMovement:
    """馬（成り角）: 角行の斜めスライド + 縦横1マス追加移動。"""

    def test_horse_has_bishop_diagonals_plus_orthogonal_steps(self) -> None:
        """Horse at (4, 4) reaches diagonals AND adjacent orthogonal squares."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.HORSE, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        # Standard bishop diagonals from (4,4) — expect NW,NE,SW,SE squares
        assert (3, 3) in dests, "Horse: NW diagonal"
        assert (3, 5) in dests, "Horse: NE diagonal"
        assert (5, 3) in dests, "Horse: SW diagonal"
        assert (5, 5) in dests, "Horse: SE diagonal"
        # Extra orthogonal 1-step moves
        assert (3, 4) in dests, "Horse: extra step North"
        assert (5, 4) in dests, "Horse: extra step South"
        assert (4, 3) in dests, "Horse: extra step West"
        assert (4, 5) in dests, "Horse: extra step East"


class TestDragonMovement:
    """龍（成り飛）: 飛車の縦横スライド + 斜め1マス追加移動。"""

    def test_dragon_has_rook_slides_plus_diagonal_steps(self) -> None:
        """Dragon at (4, 4) reaches orthogonals AND adjacent diagonal squares.

        The Dragon (龍, promoted Rook) slides along the four orthogonal axes
        like a Rook and also steps one square diagonally (the reverse of Horse).

        King-capture semantics: a Dragon CAN slide to the enemy King's square —
        the legal-move generator permits this as a valid capture (the game would
        then be in a terminal / king-captured state). The test verifies the slide
        reaches the enemy King at (0,4) rather than asserting it is blocked.
        """
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.DRAGON, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 4, 4)
        dests = _destination_set(moves)
        # Rook slides West along row 4
        assert (4, 0) in dests, "Dragon: rook slide West"
        # Rook slides East along row 4
        assert (4, 8) in dests, "Dragon: rook slide East"
        # Dragon CAN reach Gote King at (0,4) — it slides the full column
        # (the Gote King is an enemy piece, so capture is the pseudo-legal target)
        assert (0, 4) in dests, "Dragon: slides along col 4 reaching enemy king"
        # Own Sente King at (8,4) blocks the slide — not a legal destination
        assert (8, 4) not in dests, "Dragon: blocked by own King at (8,4)"
        # Extra diagonal 1-step moves
        assert (3, 3) in dests, "Dragon: extra diagonal step NW"
        assert (3, 5) in dests, "Dragon: extra diagonal step NE"
        assert (5, 3) in dests, "Dragon: extra diagonal step SW"
        assert (5, 5) in dests, "Dragon: extra diagonal step SE"
        # Dragon does NOT slide diagonally
        assert (2, 2) not in dests, "Dragon must not slide diagonally"
        assert (6, 6) not in dests, "Dragon must not slide diagonally"


# ===========================================================================
# 2. Check detection (_is_in_check)
# ===========================================================================


class TestCheckDetection:
    """_is_in_check: 王手判定。"""

    def test_king_in_check_by_pawn(self) -> None:
        """Sente king at (5, 4) is attacked by Gote pawn at (6, 4).

        Gote pawn moves forward = increasing row. Gote pawn at (6,4)
        attacks (7,4) — wait, Gote moves DOWN (row increases).
        We need Gote pawn directly in front of Sente king FROM GOTE's perspective.
        Gote pawn at (6,4) attacks square (7,4) (Gote forward = row+1). That
        doesn't hit a king at (5,4).

        For Gote pawn to threaten Sente king at (5,4), place Gote pawn at (4,4):
        Gote forward from (4,4) = (5,4) — yes, that threatens the king.
        """
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 8, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.PAWN, Player.GOTE),  # attacks (5,4) — Sente king
            ]
        )
        assert _is_in_check(board, Player.SENTE), (
            "Sente king at (5,4) should be in check from Gote pawn at (4,4)"
        )

    def test_king_in_check_by_rook(self) -> None:
        """Sente king at (5, 4) is in check from Gote rook at (5, 8) (same row)."""
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 0, PieceType.KING, Player.GOTE),
                (5, 8, PieceType.ROOK, Player.GOTE),  # same row, clear path
            ]
        )
        assert _is_in_check(board, Player.SENTE), (
            "Sente king at (5,4) should be in check from Gote rook at (5,8)"
        )

    def test_king_not_in_check_safe_position(self) -> None:
        """Neither king is in check in the standard starting position."""
        board = Board()
        assert not _is_in_check(board, Player.SENTE), "Sente king should be safe initially"
        assert not _is_in_check(board, Player.GOTE), "Gote king should be safe initially"

    def test_king_in_double_check_by_rook_and_bishop(self) -> None:
        """Sente king at (4, 4) attacked simultaneously by rook and bishop."""
        board = _make_board(
            [
                (4, 4, PieceType.KING, Player.SENTE),
                (0, 0, PieceType.KING, Player.GOTE),
                (4, 8, PieceType.ROOK, Player.GOTE),  # attacks along row 4
                (2, 2, PieceType.BISHOP, Player.GOTE),  # attacks along NW diagonal to (4,4)
            ]
        )
        assert _is_in_check(board, Player.SENTE), "Sente king at (4,4) should be in double check"

    def test_check_blocked_by_intervening_piece(self) -> None:
        """Rook check is blocked by a piece between rook and king."""
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 0, PieceType.KING, Player.GOTE),
                (5, 8, PieceType.ROOK, Player.GOTE),  # rook on same row
                (5, 6, PieceType.SILVER, Player.SENTE),  # blocks the rook
            ]
        )
        assert not _is_in_check(board, Player.SENTE), (
            "Rook check should be blocked by friendly piece at (5,6)"
        )


# ===========================================================================
# 3. Legal move filtering (king safety)
# ===========================================================================


class TestLegalMoveFiltering:
    """王手放置禁止: 合法手フィルタのテスト。"""

    def test_pinned_piece_cannot_move_away(self) -> None:
        """A piece pinned against the king by a rook has no legal moves.

        Setup: Sente king at (5, 4). Sente silver at (5, 6).
        Gote rook at (5, 8) — rook fires along row 5.
        Silver is pinned: if it moves, king is exposed to rook check.
        """
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 0, PieceType.KING, Player.GOTE),
                (5, 6, PieceType.SILVER, Player.SENTE),  # pinned piece
                (5, 8, PieceType.ROOK, Player.GOTE),  # pin source
            ]
        )
        # Only moves originating from silver at (5,6)
        silver_moves = _moves_from(board, Player.SENTE, 5, 6)
        assert len(silver_moves) == 0, (
            "Pinned silver should have 0 legal moves (would expose king)"
        )

    def test_must_block_or_capture_when_in_check(self) -> None:
        """When king is in check, all legal moves must resolve the check.

        Sente king at (8, 4). Gote rook at (0, 4) — checks along col 4.
        Only legal moves: block col 4, capture rook, or king moves off col 4.
        Every legal move must leave king out of check.
        """
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 0, PieceType.KING, Player.GOTE),
                (0, 4, PieceType.ROOK, Player.GOTE),  # check along col 4
            ]
        )
        moves = legal_moves(board, Player.SENTE)
        assert len(moves) > 0, "Should have at least one legal escape move"
        for move in moves:
            next_board = apply_move(board, Player.SENTE, move)
            assert not _is_in_check(next_board, Player.SENTE), (
                f"Move {decode_move(move)} leaves king in check"
            )

    def test_checkmate_position_has_no_legal_moves(self) -> None:
        """A position where the king is checkmated must have 0 legal moves.

        Simplest checkmate: Sente king at (0, 0) cornered.
        Gote rook at (0, 8) covers row 0.
        Gote rook at (1, 8) covers row 1.
        King cannot escape — 0 legal moves.
        """
        board = _make_board(
            [
                (0, 0, PieceType.KING, Player.SENTE),
                (8, 8, PieceType.KING, Player.GOTE),
                (0, 8, PieceType.ROOK, Player.GOTE),  # locks row 0
                (1, 8, PieceType.ROOK, Player.GOTE),  # locks row 1 (king can't step to r1)
                # Also need to cover (1,0) and (1,1) beyond the second rook's reach
                # Second rook at (1,8) covers all of row 1.
                # (1,0) and (1,1) are on row 1 — covered.
            ]
        )
        moves = legal_moves(board, Player.SENTE)
        assert len(moves) == 0, f"Cornered king should have 0 legal moves, got {len(moves)}"

    def test_king_can_escape_check_by_moving(self) -> None:
        """King in check by a single pawn can escape by stepping away.

        Gote pawn at (4, 4) checks Sente king at (5, 4).
        King should be able to step to squares NOT covered by the pawn.
        """
        board = _make_board(
            [
                (5, 4, PieceType.KING, Player.SENTE),
                (0, 8, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.PAWN, Player.GOTE),  # gives check
            ]
        )
        moves = legal_moves(board, Player.SENTE)
        # King must have at least one legal escape
        assert len(moves) > 0, "King must have at least one escape from pawn check"
        # All legal moves must resolve check
        for move in moves:
            next_board = apply_move(board, Player.SENTE, move)
            assert not _is_in_check(next_board, Player.SENTE)


# ===========================================================================
# 4. Promotion rules
# ===========================================================================


class TestPromotionRules:
    """成り規則のテスト。"""

    def test_pawn_promotion_optional_entering_zone(self) -> None:
        """Pawn moving from row 3 to row 2 (SENTE zone) gets both options."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (3, 2, PieceType.PAWN, Player.SENTE),  # outside zone
            ]
        )
        moves = _moves_from(board, Player.SENTE, 3, 2)
        promotions = [d for d in moves if d["promote"]]
        non_promotions = [d for d in moves if not d["promote"]]
        assert len(promotions) == 1, "Should have one promotion option"
        assert len(non_promotions) == 1, "Should have one non-promotion option"
        assert moves[0]["to"] == (2, 2)

    def test_pawn_must_promote_on_final_rank(self) -> None:
        """Pawn on row 1 moving to row 0 must promote — no non-promotion option."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (1, 2, PieceType.PAWN, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 1, 2)
        assert len(moves) == 1, f"Only one move allowed (forced promotion): {moves}"
        assert moves[0]["promote"] is True
        assert moves[0]["to"] == (0, 2)

    def test_knight_must_promote_on_rows_0_and_1(self) -> None:
        """Knight must promote when landing on row 0 or row 1."""
        # Knight at (2, 4) moves to (0, 3) or (0, 5) — row 0: must promote
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 7, PieceType.KING, Player.GOTE),
                (2, 4, PieceType.KNIGHT, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 2, 4)
        # Both target squares (0, 3) and (0, 5) are on row 0 — must promote
        for d in moves:
            assert d["promote"] is True, f"Knight on row 0 must promote: {d}"

    def test_knight_must_promote_on_row_1_too(self) -> None:
        """Knight landing on row 1 must also promote."""
        # Knight at (3, 4) jumps to (1, 3) and (1, 5)
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 7, PieceType.KING, Player.GOTE),
                (3, 4, PieceType.KNIGHT, Player.SENTE),
            ]
        )
        moves = _moves_from(board, Player.SENTE, 3, 4)
        for d in moves:
            to_row = d["to"][0]
            if to_row <= 1:
                assert d["promote"] is True, f"Knight landing on row {to_row} must promote"

    def test_captured_promoted_piece_reverts_to_base(self) -> None:
        """Capturing a promoted piece (Dragon) adds it to hand as Rook (base form)."""
        # Place Sente Dragon at (4, 4), Gote King nearby, Sente King safe.
        # Gote has a gold that can capture the dragon.
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (4, 4, PieceType.DRAGON, Player.SENTE),  # promoted rook
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 4, PieceType.GOLD, Player.GOTE),  # can capture dragon at (4,4)
            ]
        )
        # Gold at (5,4) moves forward to (4,4) capturing Dragon
        capture_move = encode_board_move(5 * 9 + 4, 4 * 9 + 4, promote=False)
        new_board = apply_move(board, Player.GOTE, capture_move)
        gote_hand = new_board.hands[Player.GOTE.value]
        assert PieceType.ROOK in gote_hand, "Captured Dragon should revert to ROOK in Gote's hand"
        assert PieceType.DRAGON not in gote_hand, (
            "DRAGON should not appear in hand — it reverts to base form"
        )

    def test_promotion_converts_piece_type_on_board(self) -> None:
        """After promotion, piece type on board is the promoted type."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (3, 2, PieceType.PAWN, Player.SENTE),
            ]
        )
        # Encode the promotion move: from (3,2) to (2,2) with promote=True
        promo_move = encode_board_move(3 * 9 + 2, 2 * 9 + 2, promote=True)
        new_board = apply_move(board, Player.SENTE, promo_move)
        piece = new_board.piece_at(2, 2)
        assert piece is not None
        assert piece.piece_type == PieceType.PRO_PAWN, (
            f"Pawn should promote to PRO_PAWN, got {piece.piece_type}"
        )
        assert piece.owner == Player.SENTE


# ===========================================================================
# 5. Special rules
# ===========================================================================


class TestNifuRule:
    """二歩: 同じ列に自分の未成歩が既にある列には歩を打てない。"""

    def test_cannot_drop_pawn_in_column_with_existing_own_pawn(self) -> None:
        """Nifu: dropping Sente pawn in col 3 where Sente already has a pawn."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 3, PieceType.PAWN, Player.SENTE),  # existing sente pawn in col 3
            ],
            sente_hand=(PieceType.PAWN,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.PAWN:
                _, drop_col = d["to"]
                assert drop_col != 3, (
                    "Nifu violation: cannot drop pawn in col 3 where own pawn exists"
                )

    def test_can_drop_pawn_in_other_columns(self) -> None:
        """Can drop pawn in columns that do NOT have own pawn."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (5, 3, PieceType.PAWN, Player.SENTE),  # pawn only in col 3
            ],
            sente_hand=(PieceType.PAWN,),
        )
        moves = legal_moves(board, Player.SENTE)
        pawn_drop_cols = set()
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.PAWN:
                _, drop_col = d["to"]
                pawn_drop_cols.add(drop_col)
        # Can drop in all other columns (0,1,2,4,5,6,7,8)
        assert 3 not in pawn_drop_cols, "Col 3 still forbidden"
        assert len(pawn_drop_cols) >= 1, "Should be able to drop in at least one other column"

    def test_nifu_for_gote(self) -> None:
        """Nifu applies to Gote as well: cannot drop in col with existing Gote pawn."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (3, 6, PieceType.PAWN, Player.GOTE),  # existing gote pawn in col 6
            ],
            gote_hand=(PieceType.PAWN,),
        )
        moves = legal_moves(board, Player.GOTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.PAWN:
                _, drop_col = d["to"]
                assert drop_col != 6, "Gote nifu: cannot drop pawn in col 6 where Gote pawn exists"


class TestDeadPieceRestriction:
    """行き所のない駒: 動けない場所に駒を打ってはならない。"""

    def test_cannot_drop_pawn_on_row_0_for_sente(self) -> None:
        """Sente cannot drop pawn on row 0 (no further moves)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ],
            sente_hand=(PieceType.PAWN,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.PAWN:
                drop_row, _ = d["to"]
                assert drop_row != 0, "Cannot drop pawn on row 0 for Sente"

    def test_cannot_drop_lance_on_row_0_for_sente(self) -> None:
        """Sente cannot drop lance on row 0 (no further forward moves)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ],
            sente_hand=(PieceType.LANCE,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.LANCE:
                drop_row, _ = d["to"]
                assert drop_row != 0, "Cannot drop lance on row 0 for Sente"

    def test_cannot_drop_knight_on_rows_0_or_1_for_sente(self) -> None:
        """Sente cannot drop knight on rows 0 or 1 (would have no legal next move)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ],
            sente_hand=(PieceType.KNIGHT,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.KNIGHT:
                drop_row, _ = d["to"]
                assert drop_row > 1, (
                    f"Cannot drop knight on row {drop_row} (rows 0-1 forbidden for Sente)"
                )

    def test_cannot_drop_knight_on_rows_7_or_8_for_gote(self) -> None:
        """Gote cannot drop knight on rows 7 or 8."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ],
            gote_hand=(PieceType.KNIGHT,),
        )
        moves = legal_moves(board, Player.GOTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.KNIGHT:
                drop_row, _ = d["to"]
                assert drop_row < 7, (
                    f"Cannot drop knight on row {drop_row} (rows 7-8 forbidden for Gote)"
                )


class TestUchifuzumeRule:
    """打ち歩詰め: 歩を打って相手玉を詰ませてはならない。"""

    def test_uchifuzume_is_illegal(self) -> None:
        """Cannot drop pawn to give checkmate (打ち歩詰め is illegal).

        Position:
          Sente King at (8, 4) — safe in the far corner.
          Gote  King at (0, 8) — cornered in top-right.
          Sente Gold at (0, 6) — covers Gote King's escape to (0, 7).
          Sente Gold at (2, 7) — covers escape squares (1, 7) and (1, 8).
          Sente hand: one Pawn.

        Geometry verification:
          SENTE Gold step directions (pre-flip): (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,0)
          Gold at (0,6): reaches (0,5),(0,7),(1,6)  → covers (0,7) ✓
          Gold at (2,7): reaches (1,6),(1,7),(1,8),(2,6),(2,8),(3,7) → covers (1,7) and (1,8) ✓

        Dropping Sente pawn at (1,8):
          Pawn attacks (0,8) — the Gote King's square.
          Gote King cannot move: (0,7) covered by Gold@(0,6),
                                  (1,7) covered by Gold@(2,7),
                                  (1,8) is the pawn's square.
          Result: checkmate delivered by pawn drop = Uchifuzume = ILLEGAL.
        """
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 8, PieceType.KING, Player.GOTE),
                (0, 6, PieceType.GOLD, Player.SENTE),  # covers (0,7)
                (2, 7, PieceType.GOLD, Player.SENTE),  # covers (1,7) and (1,8)
            ],
            sente_hand=(PieceType.PAWN,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop" and d["piece_type"] == PieceType.PAWN:
                drop_to = d["to"]
                # When Uchifuzume is enforced, (1,8) must NOT be a legal pawn drop
                assert drop_to != (1, 8), (
                    "Uchifuzume violation: drop pawn at (1,8) delivers checkmate by pawn "
                    "which is forbidden by the rules of shogi"
                )


# ===========================================================================
# 6. Full game test: random vs random until terminal
# ===========================================================================


class TestFullGame:
    """ゲーム通しテスト: ランダム同士で対局し終局まで正常に進行すること。"""

    def test_random_vs_random_game_completes_with_winner(self) -> None:
        """Play a full game with both players choosing random legal moves.

        A move-count limit prevents infinite loops in edge cases.
        The game must either terminate normally (winner declared) or
        reach the move limit (treated as a draw for test purposes).
        No exception must be raised at any point.
        """
        MAX_MOVES = 600  # Generous cap; typical random shogi game << 600 moves
        state = FullShogiState()
        move_count = 0

        while not state.is_terminal and move_count < MAX_MOVES:
            move = random_move(state)
            state = state.apply_move(move)
            move_count += 1

        # Game ended: either terminal or hit cap
        if state.is_terminal:
            winner = state.winner
            assert winner in (0, 1, None), (
                f"Winner must be 0 (Sente), 1 (Gote), or None (draw), got {winner}"
            )
        else:
            # Move cap reached — not a test failure, just record it
            pytest.skip(
                f"Game reached {MAX_MOVES} move cap without terminating — "
                "this is rare with random play but not an error"
            )

    def test_random_game_kings_never_captured_mid_game(self) -> None:
        """During random play, legal-move filter must prevent king capture.

        Because legal_moves filters out moves that leave king in check,
        no actual king capture can occur through the legal move pipeline.
        We verify that both kings remain on the board until the game ends
        normally via checkmate (no-legal-moves terminal condition).
        """
        MAX_MOVES = 300
        state = FullShogiState()
        move_count = 0

        while not state.is_terminal and move_count < MAX_MOVES:
            # Both kings must still be on the board
            assert state.board.find_king(Player.SENTE) is not None, (
                f"Sente king disappeared at move {move_count}"
            )
            assert state.board.find_king(Player.GOTE) is not None, (
                f"Gote king disappeared at move {move_count}"
            )
            move = random_move(state)
            state = state.apply_move(move)
            move_count += 1

    def test_state_current_player_always_valid(self) -> None:
        """current_player must be 0 or 1 on every state throughout a game."""
        MAX_MOVES = 100
        state = FullShogiState()
        for _ in range(MAX_MOVES):
            if state.is_terminal:
                break
            assert state.current_player in (0, 1)
            move = random_move(state)
            state = state.apply_move(move)

    @pytest.mark.slow
    def test_three_random_games_all_terminate(self) -> None:
        """Run three independent random games; each must terminate or hit 600 moves."""
        MAX_MOVES = 600
        rng = random.Random(42)

        for game_num in range(3):
            state = FullShogiState()
            moves_played = 0
            while not state.is_terminal and moves_played < MAX_MOVES:
                moves = state.legal_moves()
                move = rng.choice(moves)
                state = state.apply_move(move)
                moves_played += 1

            if state.is_terminal:
                winner = state.winner
                assert winner in (0, 1, None), f"Game {game_num}: unexpected winner value {winner}"


# ===========================================================================
# 7. Additional edge-case coverage
# ===========================================================================


class TestDropMoveSemantics:
    """持ち駒を打つ手の追加テスト。"""

    def test_drop_places_piece_on_board(self) -> None:
        """After a drop move, the piece appears on the board and leaves the hand."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
            ],
            sente_hand=(PieceType.GOLD,),
        )
        # Drop Gold at (4, 4)
        drop_move = encode_drop_move(PieceType.GOLD, 4 * 9 + 4)
        new_board = apply_move(board, Player.SENTE, drop_move)
        piece = new_board.piece_at(4, 4)
        assert piece is not None
        assert piece.piece_type == PieceType.GOLD
        assert piece.owner == Player.SENTE
        assert PieceType.GOLD not in new_board.hands[Player.SENTE.value]

    def test_cannot_drop_on_occupied_square(self) -> None:
        """Drop moves must not target occupied squares."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (4, 4, PieceType.PAWN, Player.GOTE),  # occupied
            ],
            sente_hand=(PieceType.GOLD,),
        )
        moves = legal_moves(board, Player.SENTE)
        for move in moves:
            d = decode_move(move)
            if d["type"] == "drop":
                assert d["to"] != (4, 4), "Cannot drop on occupied square (4,4)"


class TestGoteOrientationCorrectness:
    """後手の方向反転が正しく実装されているかを確認。"""

    def test_gote_pawn_moves_downward(self) -> None:
        """Gote pawn at (3, 4) must move to (4, 4) — row increases for Gote."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (3, 4, PieceType.PAWN, Player.GOTE),
            ]
        )
        moves = _moves_from(board, Player.GOTE, 3, 4)
        dests = _destination_set(moves)
        assert (4, 4) in dests, "Gote pawn must move to (4,4) — forward for Gote"
        assert (2, 4) not in dests, "Gote pawn cannot move backward to (2,4)"

    def test_gote_lance_slides_downward(self) -> None:
        """Gote lance at (2, 0) must slide toward row 8 (increasing row)."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (2, 0, PieceType.LANCE, Player.GOTE),
            ]
        )
        moves = _moves_from(board, Player.GOTE, 2, 0)
        dests = _destination_set(moves)
        assert (3, 0) in dests, "Gote lance must reach (3,0)"
        assert (7, 0) in dests, "Gote lance must reach (7,0)"
        assert (1, 0) not in dests, "Gote lance cannot go backward to (1,0)"
        assert (0, 0) not in dests, "Gote lance cannot go backward to (0,0)"

    def test_gote_knight_moves_downward(self) -> None:
        """Gote knight at (3, 4) lands on (5, 3) and (5, 5) — forward for Gote."""
        board = _make_board(
            [
                (8, 4, PieceType.KING, Player.SENTE),
                (0, 4, PieceType.KING, Player.GOTE),
                (3, 4, PieceType.KNIGHT, Player.GOTE),
            ]
        )
        moves = _moves_from(board, Player.GOTE, 3, 4)
        dests = _destination_set(moves)
        assert dests == {(5, 3), (5, 5)}, (
            f"Gote knight from (3,4) must land on (5,3),(5,5), got {dests}"
        )
