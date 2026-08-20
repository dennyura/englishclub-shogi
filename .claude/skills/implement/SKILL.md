---
name: implement
description: 将棋AIの章コードをTDDサイクルで実装する。引数に章番号を指定して使う（例: /implement 4）。
disable-model-invocation: true
argument-hint: "[chapter-number]"
---

# implement — TDD実装ワークフロー

対象章: $0

## 参照ファイル

- 章→タグ対応表: [chapter-map.md](chapter-map.md)
- 将棋ルール: [shogi-rules.md](shogi-rules.md)

---

## Step 1: 現状調査

以下のプロンプトを実行して **explorer エージェント**に明示的に委譲する:

```
Use the explorer agent to investigate the current state of src/shogi_ai/ and tests/
```

調査対象:
- 何のファイルが存在するか
- どのクラス・関数が実装済みか
- Protocol（`game/protocol.py`）の定義

---

## Step 2: 対象の確認

`chapter-map.md` を参照し、対象章に対応するタグと実装対象を確認する。

```bash
# タグ時点の実装例を参照する
git show <tag>:src/shogi_ai/<target-file>
```

将棋ルールの確認が必要な場合は `shogi-rules.md` を参照する。

---

## Step 3: Red — テスト作成

`tests/` に新しいテストファイルを作成する。

```bash
uv run pytest tests/ -v  # 失敗することを確認
```

テスト名は動作を明確に表す:
```python
def test_animal_shogi_initial_moves_count():  # OK
def test_1():  # NG
```

---

## Step 4: Green — 最小実装

`src/shogi_ai/` にコードを実装する。テストが通る最小限のコードを書く。

```bash
uv run pytest tests/ -v  # パスすることを確認
```

---

## Step 5: Refactor — コード整理

- 重複コードの排除
- 型ヒントの追加（全関数に必須）
- 日本語コメントの追加（将棋ルールは `shogi-rules.md` を参照）

---

## Step 6: 完了検証

```
/verify
```

全チェックがパスしたらコミットする:

```bash
git add src/ tests/
git commit -m "feat: <実装内容の説明>"
```
