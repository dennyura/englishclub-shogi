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
    model_path="/kaggle/working/best_model_animal.pt",
)

## ライセンス

MIT License
