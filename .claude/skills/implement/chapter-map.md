# 章→タグ対応表

| 章 | 内容 | タグ | 実装対象ディレクトリ |
|---|---|---|---|
| 第4〜6章 | どうぶつしょうぎエンジン | `v0.2.0` | `src/shogi_ai/game/animal_shogi/` |
| 第7章 | ミニマックス探索 | `v0.3.0` | `src/shogi_ai/engine/minimax.py` |
| 第8〜9章 | PyTorchニューラルネット | `v0.5.0` | `src/shogi_ai/model/` |
| 第10〜12章 | MCTS・自己対局・訓練 | `v0.6.0` | `src/shogi_ai/engine/mcts.py`, `src/shogi_ai/training/` |
| 第13〜14章 | 本将棋（9×9）への拡張 | `v0.8.0` | `src/shogi_ai/game/shogi/` |
| 第15章 | Web UI（FastAPI + 観戦モード） | `main` | `src/shogi_ai/web/` |

## タグ時点のコードを確認する方法

```bash
# タグ時点のコードを参照する（変更なし）
git show v0.2.0:src/shogi_ai/game/animal_shogi/state.py

# タグ間の差分を見る
git diff v0.2.0..v0.3.0 --stat

# 元に戻す（mainブランチに切り替え）
git checkout main
```

## 補足

- 第1〜3章はコードなし（AI概念・Claude Code・環境構築）
- `v0.0.0`: プロジェクト初期構築のみ（コードなし）
- `v0.4.0`, `v0.7.0` は欠番（設計見直しで廃止）
- `main`（最新）: v1.1.0 + AI vs AI観戦モード・打ち歩詰め修正を含む完成版
