## ローカルサーバー立ち上げ
uv run uvicorn shogi_ai.web.app:app --reload

 http://127.0.0.1:8000/


## Kaggle での学習

既存モデルから再開する場合だけ、Input から `.pt` ファイルを `/kaggle/working/` にコピーします。
モデルがない場合は、`run_training()` がランダム初期化したネットワークから学習を開始し、
指定した `model_path` に初期チェックポイントを新規作成します。

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
    self_play_workers=2,
    self_play_device_ids=(0, 1),
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

## config設定（本将棋）
config = TrainLoopConfig(
    num_generations=,
    num_self_play_games=1000, # ~3000
    num_simulations=400, #~800
    arena_games=100,
    win_rate_threshold=0.55,
    buffer_size=1000000,
    samples_per_generation=64000,
    batch_size=1024,
    epochs_per_generation=1,
    max_training_hours=11,
    num_res_blocks=15,
    self_play_workers=2,
    self_play_device_ids=(0, 1),
    model_path="/kaggle/working/best_model_animal.pt",
)

FULL_SHOGI_CONFIG = NetworkConfig(
    board_h=9,
    board_w=9,
    in_channels=43,
    action_size=13689,  # ACTION_SPACE in full_shogi/moves.py と一致
    num_res_blocks=15,  # 大きいゲームなので深いネットワーク
    num_channels=192,
)

# 棋譜生成
python scripts/generate_minimax_dataset.py \
  --games 10000 \
  --depth 2 \
  --opening-moves 6 \
  --top-k 3 \
  --temperature 100 \
  --max-moves 512 \
  --workers 4 \
  --output minimax_full_shogi.pt

## ライセンス

MIT License
