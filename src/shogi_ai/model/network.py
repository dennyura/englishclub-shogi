"""AlphaZero-style dual-head neural network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shogi_ai.model.config import NetworkConfig


class ResBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN + skip connection.

    残差ブロック（ResNet の基本構成要素）。

    スキップ接続により:
    - 勾配消失問題を緩和（深いネットワークでも学習可能）
    - ネットワークが「恒等写像 + 補正」を学べるようになる
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 3×3 畳み込み（padding=1 でサイズを維持）
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)  # バッチ正規化（学習安定化）
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x  # スキップ接続用に入力を保存
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + residual)  # スキップ接続: 入力を加算してからReLU
        return out


class DualHeadNetwork(nn.Module):
    """AlphaZero-style network with policy and value heads.

    AlphaZero スタイルの双頭ニューラルネットワーク。

    構造:
    [入力テンソル（局面）]
         ↓
    [入力畳み込み層]
         ↓
    [残差ブロック × N] ← 共通のボディ（特徴抽出）
         ↓         ↓
    [方策ヘッド]  [価値ヘッド]
         ↓              ↓
    [各手の確率]  [局面の価値 -1〜+1]

    Input:  (batch, in_channels, board_h, board_w)
    Output: (policy_logits, value)
        - policy_logits: (batch, action_size) — raw logits
        - value: (batch, 1) — tanh-bounded value in [-1, 1]
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        # 入力射影: チャンネル数を in_channels → num_channels に変換
        self.input_conv = nn.Conv2d(
            config.in_channels,
            config.num_channels,
            3,
            padding=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm2d(config.num_channels)

        # 残差タワー: 特徴抽出の主要部
        self.res_blocks = nn.Sequential(
            *[ResBlock(config.num_channels) for _ in range(config.num_res_blocks)]
        )

        # 方策ヘッド: 各手の「良さ」をロジットで出力
        # 1×1 畳み込みでチャンネル数を2に削減してから全結合層へ
        self.policy_conv = nn.Conv2d(config.num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * config.board_h * config.board_w,
            config.action_size,
        )

        # 価値ヘッド: 局面の勝率を -1〜+1 で出力
        # 1×1 畳み込みでチャンネル数を1に削減してから全結合層へ
        self.value_conv = nn.Conv2d(config.num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_h * config.board_w, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # 共通ボディ: 入力層 → 残差タワー
        x = torch.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # 方策ヘッド
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # フラット化: (batch, 2*h*w)
        p = self.policy_fc(p)  # ロジット: (batch, action_size)

        # 価値ヘッド
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # フラット化: (batch, h*w)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # tanh で [-1, +1] に収める

        return p, v
