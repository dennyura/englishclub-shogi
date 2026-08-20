---
name: debug
description: エラー診断と修正方針の提示。ruff/mypy/pytest/MPS/将棋ルールのエラーに対応する。verifyが失敗したときに使う。
---

# debug — エラー診断

## Step 1: エラーの分類

### A. Lint エラー（ruff）

```bash
uv run ruff check src/ tests/ --show-source
```

エラーコードを確認して修正する。よくあるコード:

| コード | 意味 | 対処 |
|---|---|---|
| `E501` | 行が長すぎる | 99文字以内に収める |
| `F401` | 未使用 import | 削除する |
| `I001` | import の順序 | `ruff check --fix` で自動修正（実行後 `git diff` で確認） |
| `UP007` | `Optional[X]` → `X \| None` | `ruff check --fix` で自動修正（実行後 `git diff` で確認） |

---

### B. 型エラー（mypy）

```bash
uv run mypy src/ --show-error-codes
```

よくあるパターン:

| エラー | 対処 |
|---|---|
| `Missing return type annotation` | 関数に戻り値型を追加する |
| `Incompatible types in assignment` | Protocol の型が一致しているか確認する |
| `Argument 1 to "..." has incompatible type` | 引数の型を確認する |

---

### C. テスト失敗（pytest）

```bash
uv run pytest tests/ -v --tb=short
```

よくあるパターン:

- **合法手生成のバグ** → `implement/shogi-rules.md` でルールを再確認する
- **テンソル形状の不一致** → `game/protocol.py` の `action_space_size` を確認する
- **不変 State の違反** → `apply_move()` が新しい State を返しているか確認する

---

### D. MPS デバイスエラー（Apple Silicon）

```python
# デバイスの状態を確認する
import torch
print(torch.backends.mps.is_available())   # True であるべき
print(torch.backends.mps.is_built())       # True であるべき
```

よくあるエラー:

| エラー | 対処 |
|---|---|
| `MPS backend out of memory` | バッチサイズを小さくする。`torch.mps.empty_cache()` を呼ぶ |
| `Not implemented for MPS` | CPU にフォールバックする。`get_device()` の実装を確認する |

---

### E. 将棋ルール違反

`implement/shogi-rules.md` を参照してルールの実装を再確認する。

特に確認すべき点:
- ひよこの向き（先手・後手で前の方向が逆）
- 成りのタイミング（相手陣に入る / 出るとき）
- 打ち歩詰め禁止の判定順序

---

## Step 2: 修正後の確認

```
/verify
```

全チェックがパスするまで修正を繰り返す。
