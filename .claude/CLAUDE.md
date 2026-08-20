# CLAUDE.md — shogi-ai-pytorch 公開リポジトリ

## プロジェクト概要

「Claude Codeで作る将棋AI」のコンパニオンリポジトリ。
読者がバイブコーディングで構築する将棋AIの完成コード。

## 技術スタック

| 技術 | バージョン | 用途 |
|------|-----------|------|
| Python | 3.13+ | メイン言語 |
| PyTorch | 2.10+ | ニューラルネットワーク |
| uv | 0.10+ | パッケージマネージャ |
| pytest | 9.0+ | テスト |
| ruff | 0.15+ | Linter |

## コード規約

- **src レイアウト**: `src/shogi_ai/` にパッケージ配置
- **不変 State**: `apply_move()` は新しい State を返す（MCTS 用）
- **Protocol**: `typing.Protocol`（PEP 544）で共通インターフェース定義
- **フルスクラッチ**: cshogi 等の外部将棋ライブラリ不使用
- **型ヒント**: 全関数に型アノテーション必須
- **ruff**: `E`, `F`, `I`, `W`, `UP` ルール、行長 99

## デバイス戦略

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- 書籍は Apple Silicon (MPS) 前提
- CI は CPU のみ（ubuntu-latest）
- CUDA 対応は明示的にスコープ外

## ディレクトリ構造

```
src/shogi_ai/
├── game/
│   ├── protocol.py          # GameState Protocol
│   └── animal_shogi/        # どうぶつしょうぎ
├── engine/                   # AI エンジン
├── model/                    # ニューラルネットワーク
├── training/                 # 訓練パイプライン
├── web/                      # Web UI
└── cli.py                    # CLI エントリポイント
```

## テスト

```bash
uv run pytest -v              # 全テスト
uv run pytest tests/game/ -v  # ゲームロジックのみ
uv run ruff check src/ tests/ # Lint
```

## Git タグ規約

| タグ | フェーズ | 対応章 | 内容 |
|------|---------|--------|------|
| `v0.0.0` | Phase 0 | — | リポジトリ初期構築 |
| `v0.2.0` | Phase 2 | Ch.4-6 | どうぶつしょうぎエンジン |
| `v0.3.0` | Phase 3 | Ch.7 | ミニマックス探索 |
| `v0.5.0` | Phase 5 | Ch.8-9 | PyTorch ニューラルネットワーク |
| `v0.6.0` | Phase 6 | Ch.10-12 | MCTS + 自己対局 + 訓練 |
| `v0.8.0` | Phase 8 | Ch.13-14 | 本将棋への拡張 |
| `v1.0.0` | Phase 9 | Ch.15 | Web UI + 完成版 |

## 実装方針

- **全フェーズ完成後に執筆開始**: コードを先に完成させ、リファクタによる章の書き直しを防ぐ
- **コードファースト**: テスト作成 → 実装 → CI 通過 → `git tag` の順で進める
- **Protocol 拡張は後方互換**: 新プロパティ追加時、既存の実装を壊さない
  - Phase 5 で `action_space_size` プロパティを Protocol に追加
  - 既存の `AnimalShogiState` にも同時に実装する

## Phase 実装ワークフロー

```
1. テスト作成（TDD: Red）
2. 最小実装（Green）
3. リファクタ（Refactor）
4. ruff check + pytest → 全パス
5. git tag vX.Y.Z
```

## 強さ検証基準

| Phase | 検証方法 | 合格基準 | 備考 |
|-------|---------|---------|------|
| Phase 3 | vs ランダム 100局 | 勝率 >90% (depth=4) | 自動テスト |
| Phase 6 | 訓練後 vs ランダム 50局 | 勝率 >80% | arena.py |
| Phase 6 | 訓練後 vs minimax(depth=2) | 勝率 >50% | arena.py |
| Phase 8 | 初期局面の合法手数 | 30手 | 自動テスト |
| Phase 9 | Playwright MCP 実対局 | 対局完了可能 | 手動検証 |

## Protocol 拡張計画

| Phase | 変更 | 理由 |
|-------|------|------|
| Phase 5 | `action_space_size: int` プロパティ追加 | NN 出力サイズに必要 |
| Phase 8 | 変更なし（Config で対応） | `DualHeadNetwork` は Config で board_h/w/action_size を受ける |

## 利用可能なスキル

Claude Code でリポジトリを開くと以下のスキルが使えます。

| スキル | 呼び出し方 | 内容 |
|--------|-----------|------|
| `implement` | `/implement 4` | 章コードをTDDで実装（explorer エージェントと連携） |
| `verify` | `/verify` | ruff・mypy・pytest で完了検証 |
| `test-e2e` | `/test-e2e` | Playwright MCPでWeb UIを実対局テスト |
| `debug` | `/debug` | エラー診断と修正方針の提示 |

### 標準的な使い方

```
/implement 4   # 第4章のコードを実装する
/verify        # 実装完了を検証する
/implement 7   # 第7章へ進む
...
/test-e2e      # 第15章完了後にWeb UIをテストする
```

### 利用可能なエージェント

- **explorer**: コード調査専門。`implement` から自動で委譲される。`/agents` で確認できる。
