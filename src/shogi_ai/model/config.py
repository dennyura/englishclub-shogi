"""Network configuration for dual-head AlphaZero-style models.

ニューラルネットワークの設定定義。
ゲームによって盤面サイズ・チャンネル数・行動数が異なるため、設定クラスで管理する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for DualHeadNetwork.

    DualHeadNetwork の設定パラメータ。

    Attributes:
        board_h:      盤面の高さ（行数）
        board_w:      盤面の幅（列数）
        in_channels:  入力特徴プレーン数（局面のテンソル表現のチャンネル数）
        action_size:  行動空間のサイズ（合法手の最大数）
        num_res_blocks: 残差ブロックの数（多いほど表現力が高いが学習が重い）
        num_channels:   畳み込み層のチャンネル数（多いほど表現力が高い）
    """

    board_h: int
    board_w: int
    in_channels: int
    action_size: int
    num_res_blocks: int = 3
    num_channels: int = 64


# どうぶつしょうぎ用のプリセット設定
# 盤面: 3×4、入力: 14チャンネル（5駒種×2プレイヤー + 持ち駒3種 + 手番1）
# 行動空間: 180（盤上の手144 + 打ち手36）
ANIMAL_SHOGI_CONFIG = NetworkConfig(
    board_h=4,
    board_w=3,
    in_channels=14,
    action_size=180,
    num_res_blocks=3,  # 小さいゲームなので浅いネットワーク
    num_channels=64,
)

# 本将棋用のプリセット設定
# 盤面: 9×9、入力: 43チャンネル（14駒種×2 + 持ち駒7種×2 + 手番1）
# 行動空間: 13689（盤上手13122 + 打ち手567）
# MCTSの _evaluate() が policy[m] で直接インデックスするため
# action_size は全手エンコードの最大値以上でなければならない
FULL_SHOGI_CONFIG = NetworkConfig(
    board_h=9,
    board_w=9,
    in_channels=43,
    action_size=13689,  # ACTION_SPACE in full_shogi/moves.py と一致
    num_res_blocks=5,  # 大きいゲームなので深いネットワーク
    num_channels=128,
)
