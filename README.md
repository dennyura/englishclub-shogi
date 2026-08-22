## ローカルサーバー立ち上げ
uv run uvicorn shogi_ai.web.app:app --reload

 http://127.0.0.1:8000/

 import shutil

src_file = "/kaggle/input/datasets/dennyura/englishclub-shogi/example.txt"
dst_file = "/kaggle/working/example.txt"

shutil.copy(src_file, dst_file)



## Kaggle から Google Drive へ保存

Kaggle の `/kaggle/working/` はセッション終了時に削除されるため、Google Drive API を使って
モデルを同期します。Kaggle の **Add-ons > Secrets** に、サービスアカウント JSON 全体を
`DRIVE_SERVICE_ACCOUNT_JSON` という名前で登録してください。Google Drive の保存先フォルダは
サービスアカウントのメールアドレス（例: `shogiai@kaggle-shogi.iam.gserviceaccount.com`）へ
編集者として共有し、フォルダ URL の ID を `DRIVE_FOLDER_ID` に設定します。

次のセルを学習セルより前に一度実行します。

```python
!pip install -q google-api-python-client google-auth
```

```python
import shutil
from pathlib import Path

from shogi_ai.training.drive_sync import GoogleDriveSync

DRIVE_FOLDER_ID = "Google DriveフォルダのID"
MODEL_NAME = "best_model_animal.pt"
LOCAL_MODEL_PATH = Path("/kaggle/working") / MODEL_NAME

drive_sync = GoogleDriveSync.from_kaggle_secret(
    "DRIVE_SERVICE_ACCOUNT_JSON",
    DRIVE_FOLDER_ID,
)

# Drive に既存モデルがあれば、学習開始前に working へ取得する。
drive_sync.download_checkpoint(LOCAL_MODEL_PATH, MODEL_NAME)
```

```python
from shogi_ai.training.train_loop import TrainLoopConfig, run_training

config = TrainLoopConfig(
    num_generations=100,
    num_self_play_games=100,
    num_simulations=100,
    arena_games=40,
    max_training_hours=8,
    self_play_workers=2,
    self_play_device_ids=(0, 1),
    model_path=str(LOCAL_MODEL_PATH),
    # 採用された世代の直後に Drive 上の .pt を上書きする。
    checkpoint_callback=drive_sync.upload_checkpoint,
    # 全世代が正常終了した時に Drive の log.txt へ追記する。
    log_callback=drive_sync.append_log,
)

run_training(
    AnimalShogiState(),
    ANIMAL_SHOGI_CONFIG,
    config,
    progress_queue,
    stop_event,
)
```

`run_training()` は起動時に `model_path` のファイルを読み込み、採用時だけ
`checkpoint_callback` を呼び出します。全学習が正常終了した時だけ `log_callback` を呼び出し、
Drive 上の既存 `log.txt` を残したまま今回の世代ログを末尾へ追加します。途中停止や例外時にも
ログを残したい場合は、Kaggle の出力ログを別途保存してください。

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
    model_path="/content/drive/MyDrive/shogi-ai/best_model_animal.pt",
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
