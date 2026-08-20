---
name: verify
description: ruff・mypy・pytestで実装の完了を検証する。implementのStep 6として使う。問題があれば/debugを案内する。
---

# verify — 実装完了検証

## Step 1: Lint

```bash
uv run ruff check src/ tests/
```

エラー 0 を確認する。

---

## Step 2: 型チェック

```bash
uv run mypy src/
```

`error:` の行数が 0 であることを確認する。
`note:` や `warning:` のみであれば問題なし（`strict = false` モードで実行）。

---

## Step 3: テスト

```bash
uv run pytest -v
```

全テストがパスすることを確認する。

---

## Step 4: 結果報告

| 条件 | 結果 |
|---|---|
| 全チェックパス | 「タグ打ち可能」と報告する |
| 失敗あり | 失敗内容を報告し、`/debug` を案内する |
