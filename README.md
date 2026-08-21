## ローカルサーバー立ち上げ
uv run uvicorn shogi_ai.web.app:app --reload

 http://127.0.0.1:8000/


import shutil

# コピー元（Input）とコピー先（Working）のパスを指定
src = "/kaggle/input/datasets/dennyura/englishclub-shogi/best_model_animal.pt"
dst = "/kaggle/working"

shutil.copy("/kaggle/input/datasets/dennyura/englishclub-shogi/best_model_animal.pt", "/kaggle/working/")

## config設定（どうぶつ将棋）

config = TrainLoopConfig(
    num_generations=10,
    num_self_play_games=100,
    num_simulations=100,
    arena_games=40,
    win_rate_threshold=0.55,
    buffer_size=30000,
    samples_per_generation=3000,
    batch_size=128,
    epochs_per_generation=5,
    max_training_hours=8,
    num_res_blocks=5,
    model_path="/kaggle/working/best_model_animal.pt",
)

FULL_SHOGI_CONFIG = NetworkConfig(
    board_h=9,
    board_w=9,
    in_channels=43,
    action_size=13689,  # ACTION_SPACE in full_shogi/moves.py と一致
    num_res_blocks=10,  # 大きいゲームなので深いネットワーク
    num_channels=128,
)

## configの目安

パラメータ,どうぶつ将棋,本将棋（個人〜中規模環境）,変更の理由
num_generations,10 〜 50,"1,000 〜 10,000+",局面のパターンが膨大なため、長期的な学習が必要です。
num_self_play_games,100,"1,000 〜 5,000",1世代で生成すべきデータの必要量が桁違いに増えます。
num_simulations,100,400 〜 800,本将棋の複雑な中終盤を見極めるには、AlphaZero同等（800）の探索深度が必要です。
arena_games,40,100 〜 200,勝率評価のブレ（先後差や偶然の勝ち）を抑えるために増やします。
buffer_size,"30,000","1,000,000 〜 5,000,000",序盤・中盤・終盤の多様な局面を記憶するために超大容量が必要です。
samples_per_generation,"3,000","50,000 〜 200,000",1世代ごとの更新に必要なサンプル局面数です。
batch_size,128,"1,024 〜 4,096",深いResNetモデルと大容量データをGPUの並列処理で効率よく回します。
epochs_per_generation,5,1 〜 2,データ量が膨大なため、過学習を防ぐ目的で周回数は小さくします。

## ライセンス

MIT License
