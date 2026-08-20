"""Monte Carlo Tree Search (MCTS) with neural network guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from shogi_ai.game.protocol import GameState
from shogi_ai.model.network import DualHeadNetwork


@dataclass
class MCTSNode:
    """A node in the MCTS tree.

    MCTSの探索木の1ノード。各ノードは1つの局面に対応する。

    visit_count (N): このノードが訪問された回数
    total_value (W): このノードを通じた対局結果の合計値
    prior (P):       ニューラルネットワークが推定した事前確率
    children:        子ノード（合法手 → 次局面）の辞書
    """

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Average value (W/N).

        Q値 = 総価値 / 訪問回数。
        このノードを通じた対局の平均結果（+1=勝, -1=負, 0=引き分け）。
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 50  # 1手あたりのシミュレーション回数
    c_puct: float = 1.4  # 探索と活用のバランス係数（大きいほど探索重視）
    temperature: float = 1.0  # 行動選択の温度（高いほど探索的）
    dirichlet_alpha: float = 0.3  # ディリクレノイズの集中度パラメータ
    dirichlet_epsilon: float = 0.25  # ノイズの混合率（25%をノイズに）


class MCTS:
    """Monte Carlo Tree Search guided by a neural network.

    AlphaZero スタイルの MCTS。
    ニューラルネットワークが各局面を評価し、探索を効率的に誘導する。

    アルゴリズムの4ステップ:
    1. 選択 (Selection):   PUCT スコアで子ノードを選ぶ
    2. 展開 (Expansion):   葉ノードをニューラルネットで評価・展開
    3. バックアップ (Backup): 評価値を根ノードまで伝播
    4. 行動選択 (Action):  訪問回数に基づいて確率を計算
    """

    def __init__(self, network: DualHeadNetwork, config: MCTSConfig) -> None:
        self.network = network
        self.config = config
        # ニューラルネットの計算デバイス（CPU or MPS/GPU）
        self.device = next(network.parameters()).device

    def search(self, state: GameState) -> list[float]:
        """Run MCTS and return action probabilities.

        MCTSを実行し、各行動の選択確率を返す。

        Returns a list of length action_space_size with visit-count-based
        probabilities for each action.
        """
        root = MCTSNode()
        legal = state.legal_moves()

        if not legal:
            return [0.0] * state.action_space_size

        # ルートノードをニューラルネットで評価・展開
        policy, _ = self._evaluate(state)
        for move in legal:
            root.children[move] = MCTSNode(prior=policy[move])

        # ルートにディリクレノイズを加えて探索の多様性を確保
        # （同じ局面から毎回同じ手を選ばないようにする）
        self._add_dirichlet_noise(root, legal)

        # num_simulations 回のシミュレーションを実行
        for _ in range(self.config.num_simulations):
            self._simulate(root, state)

        # 訪問回数から行動確率を計算
        action_probs = [0.0] * state.action_space_size
        if self.config.temperature == 0:
            # 温度0: 最も訪問されたノードを決定論的に選択（本番対局用）
            best_move = max(root.children, key=lambda m: root.children[m].visit_count)
            action_probs[best_move] = 1.0
        else:
            # 温度パラメータで訪問回数を変換して確率を計算
            # temperature が小さいほど最多訪問手を重視する
            total = sum(
                child.visit_count ** (1.0 / self.config.temperature)
                for child in root.children.values()
            )
            if total > 0:
                for move, child in root.children.items():
                    action_probs[move] = (
                        child.visit_count ** (1.0 / self.config.temperature) / total
                    )

        return action_probs

    def _simulate(self, node: MCTSNode, state: GameState) -> float:
        """Run one simulation from the given node/state. Returns value.

        1回のシミュレーション（選択→展開→バックアップ）。
        戻り値: 現在のプレイヤー視点の価値（+1=勝, -1=負, 0=引き分け）
        """
        # 終局状態: 実際の結果を返す
        if state.is_terminal:
            if state.winner is None:
                return 0.0
            if state.winner == state.current_player:
                return 1.0
            return -1.0

        # 葉ノード（未展開）: ニューラルネットで評価して展開
        if not node.children:
            policy, value = self._evaluate(state)
            legal = state.legal_moves()
            for move in legal:
                node.children[move] = MCTSNode(prior=policy[move])
            # ニューラルネットの価値推定をそのまま使う（ロールアウト不要）
            return value

        # 内部ノード: PUCT スコアで子ノードを選択
        move = self._select_child(node)
        child = node.children[move]
        next_state = state.apply_move(move)

        # 再帰的にシミュレーション（相手番なので符号反転）
        value = -self._simulate(child, next_state)

        # バックアップ: 訪問回数と総価値を更新
        child.visit_count += 1
        child.total_value += value

        return value

    def _select_child(self, node: MCTSNode) -> int:
        """Select child with highest PUCT score.

        PUCT (Predictor + UCT) スコアで子ノードを選ぶ。

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Q(s,a): これまでの平均価値（活用）
        P(s,a): ニューラルネットの事前確率（誘導）
        N(s):   親の訪問回数（探索ボーナスのスケール）
        N(s,a): この子の訪問回数（未探索ほどボーナスが大きい）
        """
        # 親の訪問回数の平方根（探索ボーナスのスケーリング用）
        parent_n = sum(c.visit_count for c in node.children.values())
        sqrt_parent = math.sqrt(parent_n + 1)

        best_move = -1
        best_score = float("-inf")

        for move, child in node.children.items():
            # PUCT スコア = 活用（Q値）+ 探索ボーナス
            puct = child.q_value + self.config.c_puct * child.prior * sqrt_parent / (
                1 + child.visit_count
            )
            if puct > best_score:
                best_score = puct
                best_move = move

        return best_move

    def _evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Evaluate a state with the neural network.

        ニューラルネットで局面を評価する。

        Returns (policy_probs, value) where policy_probs is masked to legal moves.

        policy_probs: 各行動の選択確率（合法手のみ、ソフトマックス適用済み）
        value:        局面の価値（+1=現プレイヤー勝利, -1=敗北）
        """
        # 局面をテンソルに変換してニューラルネットに入力
        tensor = state.to_tensor_planes().unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():  # 勾配計算不要（推論のみ）
            policy_logits, value_tensor = self.network(tensor)

        legal = state.legal_moves()
        policy = policy_logits[0].cpu()

        # 違法手のロジットを -inf にして確率をゼロにマスク
        mask = torch.full_like(policy, float("-inf"))
        for m in legal:
            mask[m] = policy[m]

        # ソフトマックスで確率分布に変換
        probs = torch.softmax(mask, dim=0).tolist()
        value = value_tensor.item()

        return probs, value

    def _add_dirichlet_noise(
        self,
        root: MCTSNode,
        legal_moves: list[int],
    ) -> None:
        """Add Dirichlet noise to root priors for exploration.

        ルートノードの事前確率にディリクレノイズを加える。
        これにより、自己対局中に同じ局面でも異なる手を試せる。

        混合: new_prior = (1 - ε) * P(s,a) + ε * noise
        ε=0.25 → 事前確率の25%をランダムノイズに置き換える
        """
        noise = (
            torch.distributions.Dirichlet(
                torch.full((len(legal_moves),), self.config.dirichlet_alpha)
            )
            .sample()
            .tolist()
        )

        eps = self.config.dirichlet_epsilon
        for i, move in enumerate(legal_moves):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * noise[i]
