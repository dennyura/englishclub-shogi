"""Training loop for AlphaZero-style learning."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.self_play import TrainingExample


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the training loop."""

    lr: float = 1e-3  # 学習率（Adam オプティマイザ）
    weight_decay: float = 1e-4  # L2 正則化係数（過学習防止）
    batch_size: int = 64  # ミニバッチサイズ
    epochs_per_generation: int = 10  # 1世代あたりのエポック数
    buffer_size: int = 10000  # リプレイバッファの最大サイズ


class Trainer:
    """Train the dual-head network from self-play data.

    自己対局データからニューラルネットワークを訓練するクラス。

    AlphaZero の損失関数:
    L = L_policy + L_value

    L_policy: 方策損失（クロスエントロピー）
        MCTS の訪問回数分布を教師に、ニューラルネットの方策を近づける。

    L_value: 価値損失（平均二乗誤差）
        対局結果（±1）を教師に、ニューラルネットの価値推定を近づける。
    """

    def __init__(
        self,
        network: DualHeadNetwork,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.network = network
        self.config = config
        self.device = device
        # Adam オプティマイザ（SGD より安定して学習しやすい）
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,  # L2 正則化
        )

    def train(self, examples: list[TrainingExample]) -> dict[str, float]:
        """Train for one generation. Returns average losses.

        1世代分の訓練を行い、平均損失を返す。

        1世代 = 自己対局データを epochs_per_generation 回繰り返してミニバッチ学習
        """
        if not examples:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        self.network.train()  # 訓練モード（バッチ正規化・ドロップアウトが有効）
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        for _ in range(self.config.epochs_per_generation):
            # エポックごとにシャッフル（過学習防止・勾配の偏り解消）
            random.shuffle(examples)
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                if not batch:
                    continue

                # テンソルをまとめてデバイスに送る
                states = torch.stack([ex.state_tensor for ex in batch]).to(self.device)
                target_policies = torch.stack([ex.policy_target for ex in batch]).to(self.device)
                target_values = (
                    torch.tensor(
                        [ex.value_target for ex in batch],
                        dtype=torch.float32,
                    )
                    .unsqueeze(1)
                    .to(self.device)
                )

                # 順伝播（フォワードパス）
                policy_logits, values = self.network(states)

                # 方策損失: クロスエントロピー（MCTS確率分布との差）
                # log_softmax + 内積 でクロスエントロピーを効率的に計算
                log_probs = nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

                # 価値損失: 平均二乗誤差（対局結果との差）
                value_loss = nn.functional.mse_loss(values, target_values)

                # 合計損失（方策損失 + 価値損失）
                loss = policy_loss + value_loss

                # 逆伝播と重み更新
                self.optimizer.zero_grad()  # 勾配をリセット
                loss.backward()  # 逆伝播で勾配計算
                self.optimizer.step()  # オプティマイザで重み更新

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batches += 1

        if total_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # バッチ数で割って平均損失を返す
        avg_policy = total_policy_loss / total_batches
        avg_value = total_value_loss / total_batches
        return {
            "policy_loss": avg_policy,
            "value_loss": avg_value,
            "total_loss": avg_policy + avg_value,
        }
