"""Edge case tests for MCTS search.

MCTS のエッジケーステスト。
最小シミュレーション数や終局直前局面での動作を確認する。
"""

from __future__ import annotations

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.moves import ACTION_SPACE, encode_board_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import COLS, PieceType, Player
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork


def _make_network() -> DualHeadNetwork:
    net = DualHeadNetwork(ANIMAL_SHOGI_CONFIG)
    net.eval()
    return net


def _make_state(squares: list[Piece | None], player: Player = Player.SENTE) -> AnimalShogiState:
    board = Board(squares=tuple(squares), hands=((), ()))
    return AnimalShogiState(board=board, _current_player=player)


class TestMCTSMinimalSimulations:
    def test_num_simulations_1(self) -> None:
        """num_simulations=1 の最小ケースで正しい確率分布を返す。"""
        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=1))
        state = AnimalShogiState()
        probs = mcts.search(state)

        assert len(probs) == ACTION_SPACE
        # 確率の合計は 1（または終局なら 0）
        total = sum(probs)
        assert abs(total - 1.0) < 0.01

    def test_probabilities_sum_to_one(self) -> None:
        """通常局面では確率の合計が 1.0 になる。"""
        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=5))
        state = AnimalShogiState()
        probs = mcts.search(state)
        assert abs(sum(probs) - 1.0) < 0.01

    def test_illegal_moves_have_zero_probability(self) -> None:
        """非合法手の確率は 0 であること。"""
        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=5))
        state = AnimalShogiState()
        probs = mcts.search(state)
        legal = set(state.legal_moves())
        for i, p in enumerate(probs):
            if i not in legal:
                assert p == 0.0 or abs(p) < 1e-6, f"非合法手 {i} に確率 {p} が割り当てられた"


class TestMCTSNearTerminal:
    def test_one_move_from_win(self) -> None:
        """ライオン取りが1手で可能な局面で、MCTSがその手を高確率で選ぶ。"""
        squares: list[Piece | None] = [None] * 12
        # 後手ライオン (0,1) / 先手ライオン (3,1) / 先手きりん (1,1) で取れる
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        squares[1 * COLS + 1] = Piece(PieceType.GIRAFFE, Player.SENTE)
        state = _make_state(squares, Player.SENTE)
        assert not state.is_terminal  # まだ終局ではない

        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=30))
        probs = mcts.search(state)

        winning_move = encode_board_move(1 * COLS + 1, 0 * COLS + 1)
        # 勝ち手が最も高い確率を持つべき
        assert probs[winning_move] == max(probs)

    def test_terminal_state_returns_all_zeros(self) -> None:
        """終局状態では全手の確率が 0 になる（合法手なし）。"""
        squares: list[Piece | None] = [None] * 12
        squares[10] = Piece(PieceType.LION, Player.SENTE)  # 先手ライオンのみ
        state = _make_state(squares, Player.SENTE)
        assert state.is_terminal

        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=5))
        probs = mcts.search(state)
        assert sum(probs) == 0.0

    def test_already_won_state(self) -> None:
        """勝敗確定済み局面でも MCTS がクラッシュしない。"""
        squares: list[Piece | None] = [None] * 12
        # 後手ライオンなし → 先手の勝ち（is_terminal=True）
        squares[10] = Piece(PieceType.LION, Player.SENTE)
        state = _make_state(squares, Player.GOTE)
        assert state.is_terminal
        assert state.winner == Player.SENTE.value

        net = _make_network()
        mcts = MCTS(net, MCTSConfig(num_simulations=1))
        probs = mcts.search(state)
        # クラッシュせず、全ゼロを返す
        assert sum(probs) == 0.0
