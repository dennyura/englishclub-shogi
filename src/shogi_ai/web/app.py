"""FastAPI web application for playing shogi against AI.

FastAPI を使った将棋AI Web アプリケーション。
ブラウザから将棋AIと対戦できる REST API を提供する。

エンドポイント:
  GET  /              — フロントエンドの HTML を返す
  POST /api/new-game  — 新規対局を開始（ゲームIDを返す）
  POST /api/move      — プレイヤーが手を指す（AIが応答して次局面を返す）
  GET  /api/state/{id} — 現在の局面情報を取得
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.engine.minimax import minimax_move
from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.display import board_to_str as animal_format
from shogi_ai.game.animal_shogi.moves import decode_move as animal_decode
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.full_shogi.display import format_board as full_format
from shogi_ai.game.full_shogi.moves import decode_move as full_decode
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG, FULL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.train_loop import TrainLoopConfig, run_training

# 静的ファイル（HTML, CSS, JS）のディレクトリ
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Shogi AI")
# /static/ 以下で静的ファイルを配信
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 対局情報のインメモリストレージ（サーバ再起動で消える簡易実装）
# 本番環境では Redis や DB に保存する
_games: dict[str, dict[str, Any]] = {}

# 訓練セッション管理（同時に1セッションのみ）
_train_state: dict[str, Any] = {
    "running": False,
    "stop_event": None,
    "thread": None,
    "progress_queue": None,
}

class NewGameRequest(BaseModel):
    """新規対局リクエストのスキーマ。"""

    game_type: str = "animal"  # "animal"（どうぶつしょうぎ）or "full"（本将棋）
    ai_type: str = "minimax"  # 後手のAI種別: "minimax", "random", "mcts"
    sente_type: str = "human"  # 先手の種別: "human" or AI種別（AI同士の観戦モード）


class MoveRequest(BaseModel):
    """指し手リクエストのスキーマ。"""

    game_id: str  # 対局ID（/api/new-game で取得）
    move: int  # 手のエンコード値（整数）


class TrainRequest(BaseModel):
    """訓練開始リクエストのスキーマ。"""

    game_type: str = "animal"  # "animal" or "full"
    num_generations: int = 10  # 訓練する世代数


def _get_ai_fn(
    ai_type: str,
    game_type: str,
) -> Callable[[GameState], int]:
    """Get the AI move function based on type.

    AI種別に応じた手選択関数を返す。
    ai_type="mcts" かつ訓練済みモデルが読み込まれている場合は、
    そのモデルを使用して MCTS を実行する。
    """
    if ai_type == "random":
        # ランダムAI（最弱、動作確認用）
        return lambda state: random_move(state)
    if ai_type == "minimax":
        # ミニマックスAI: どうぶつは深さ4、本将棋は深さ2（計算時間の兼ね合い）
        depth = 4 if game_type == "animal" else 2
        return lambda state: minimax_move(state, depth=depth)
    if ai_type == "mcts":
        # ゲーム種別に応じたネットワーク設定を選択
        config = ANIMAL_SHOGI_CONFIG if game_type == "animal" else FULL_SHOGI_CONFIG
        net = DualHeadNetwork(config)
        model_path = Path(f"best_model_{game_type}.pt")
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"{model_path} が見つからないため、MCTS AI との対局を開始できません。",
            )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()  # 推論モード
        mcts = MCTS(net, MCTSConfig(num_simulations=50))  # 1手あたりのシミュレーション回数を増やすと強くなる

        def mcts_move(state: GameState) -> int:
            probs = mcts.search(state)
            legal = state.legal_moves()
            # 最も確率の高い合法手を選択
            return max(legal, key=lambda m: probs[m])

        return mcts_move
    msg = f"Unknown AI type: {ai_type}"
    raise ValueError(msg)


def _state_to_dict(state: GameState, game_type: str) -> dict[str, Any]:
    """Convert game state to JSON-serializable dict.

    局面情報を JSON 形式（辞書）に変換する。
    フロントエンドの JavaScript がこの形式を受け取って盤面を描画する。
    """
    board = state.board  # type: ignore[attr-defined]

    if game_type == "animal":
        board_display = animal_format(board)
        squares: list[dict[str, Any] | None] = []
        for piece in board.squares:
            if piece is None:
                squares.append(None)
            else:
                squares.append(
                    {
                        "type": piece.piece_type.value,  # 駒種インデックス
                        "owner": piece.owner.value,  # 所有者（0=先手, 1=後手）
                        "name": piece.piece_type.name,  # 駒名（文字列）
                    }
                )
        hands = [
            [pt.name for pt in board.hands[0]],  # 先手の持ち駒
            [pt.name for pt in board.hands[1]],  # 後手の持ち駒
        ]
        rows, cols = 4, 3
    else:
        board_display = full_format(board)
        squares = []
        for piece in board.squares:
            if piece is None:
                squares.append(None)
            else:
                squares.append(
                    {
                        "type": piece.piece_type.value,
                        "owner": piece.owner.value,
                        "name": piece.piece_type.name,
                    }
                )
        hands = [
            [pt.name for pt in board.hands[0]],
            [pt.name for pt in board.hands[1]],
        ]
        rows, cols = 9, 9

    return {
        "current_player": state.current_player,  # 手番（0=先手, 1=後手）
        "is_terminal": state.is_terminal,  # 終局フラグ
        "winner": state.winner,  # 勝者（None=対局中）
        "legal_moves": state.legal_moves(),  # 合法手リスト
        "squares": squares,  # 盤面の駒情報（81または12要素）
        "hands": hands,  # 持ち駒情報
        "rows": rows,
        "cols": cols,
        "board_display": board_display,  # テキスト形式の盤面表示
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """フロントエンドの HTML を配信する。"""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/api/new-game")
async def new_game(req: NewGameRequest) -> dict[str, Any]:
    """新規対局を開始する。

    対局IDと初期局面情報を返す。
    対局IDはその後の手番送信（/api/move）で使用する。
    """
    game_id = str(uuid.uuid4())[:8]  # 短いIDを生成

    # ゲーム種別に応じた初期局面を作成
    if req.game_type == "animal":
        state: GameState = AnimalShogiState()
    elif req.game_type == "full":
        state = FullShogiState()
    else:
        raise HTTPException(400, f"Unknown game type: {req.game_type}")

    # 先手・後手の手選択関数を初期化
    # sente_type="human" なら先手は人間（None = クリック操作）
    sente_fn = None if req.sente_type == "human" else _get_ai_fn(req.sente_type, req.game_type)
    gote_fn = _get_ai_fn(req.ai_type, req.game_type)

    # 対局情報をメモリに保存
    _games[game_id] = {
        "state": state,
        "game_type": req.game_type,
        "sente_fn": sente_fn,  # None = 人間（先手）
        "gote_fn": gote_fn,  # 後手は常にAI
        "ai_fn": gote_fn,  # 後方互換: /api/move で使用
    }

    return {
        "game_id": game_id,
        "state": _state_to_dict(state, req.game_type),
    }


@app.post("/api/move")
async def make_move(req: MoveRequest) -> dict[str, Any]:
    """プレイヤーの手を受け取り、AIが応答して次の局面を返す。

    処理フロー:
    1. プレイヤーの手を検証して適用
    2. AI がミニマックス/MCTS で最善手を計算
    3. AI の手を適用して新局面を返す
    """
    game = _games.get(req.game_id)
    if game is None:
        raise HTTPException(404, "Game not found")

    state: GameState = game["state"]
    game_type: str = game["game_type"]

    if state.is_terminal:
        raise HTTPException(400, "Game is already over")

    if req.move not in state.legal_moves():
        raise HTTPException(400, f"Illegal move: {req.move}")

    # プレイヤーの手を適用
    state = state.apply_move(req.move)

    # ゲームが終わっていなければ AI が応答
    ai_move = None
    if not state.is_terminal:
        ai_fn = game["ai_fn"]
        ai_move = ai_fn(state)
        state = state.apply_move(ai_move)

    # 最新の局面を保存
    game["state"] = state

    decode_fn = animal_decode if game_type == "animal" else full_decode

    return {
        "state": _state_to_dict(state, game_type),
        "player_move": req.move,
        "ai_move": ai_move,
        "ai_move_decoded": decode_fn(ai_move) if ai_move is not None else None,
    }


@app.get("/api/state/{game_id}")
async def get_state(game_id: str) -> dict[str, Any]:
    """現在の局面情報を取得する（ページ再読み込み時などに使用）。"""
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, "Game not found")
    return _state_to_dict(game["state"], game["game_type"])


@app.post("/api/train/start")
async def train_start(req: TrainRequest) -> dict[str, Any]:
    """訓練を開始する。

    バックグラウンドスレッドで AlphaZero 訓練ループを起動する。
    進捗は /api/train/stream (SSE) で取得できる。
    """
    if _train_state["running"]:
        raise HTTPException(400, "Training is already running")

    if req.game_type == "animal":
        state: GameState = AnimalShogiState()
        net_config = ANIMAL_SHOGI_CONFIG
        model_path = "best_model_animal.pt"
    elif req.game_type == "full":
        state = FullShogiState()
        net_config = FULL_SHOGI_CONFIG
        model_path = "best_model_full.pt"
    else:
        raise HTTPException(400, f"Unknown game type: {req.game_type}")

    loop_config = TrainLoopConfig(
        num_generations=req.num_generations,
        model_path=model_path,
    )

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=run_training,
        args=(state, net_config, loop_config, progress_queue, stop_event),
        daemon=True,
    )

    _train_state["running"] = True
    _train_state["stop_event"] = stop_event
    _train_state["thread"] = thread
    _train_state["progress_queue"] = progress_queue
    _train_state["game_type"] = req.game_type

    thread.start()
    return {
        "status": "started",
        "game_type": req.game_type,
        "num_generations": req.num_generations,
    }


@app.get("/api/train/stream")
async def train_stream() -> StreamingResponse:
    """SSE で訓練の進捗をリアルタイム配信する。

    Server-Sent Events (SSE) 形式でイベントを送信する。
    ブラウザ側は EventSource で受信する。

    イベント種別:
      phase:           フェーズ開始（self_play / training / arena）
      generation_done: 1世代完了（損失・勝率・採用結果を含む）
      done:            全世代完了
      stopped:         ユーザーが停止
      heartbeat:       接続維持（30秒ごと）
    """
    q: queue.Queue[dict[str, Any]] | None = _train_state.get("progress_queue")
    if q is None:
        raise HTTPException(404, "No training session")

    async def event_generator() -> Any:
        loop = asyncio.get_running_loop()
        while True:
            try:
                # 最大1秒待機（スレッドプールでブロッキング get を実行）
                event = await loop.run_in_executor(None, lambda: q.get(timeout=1))
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "stopped"):
                    _train_state["running"] = False
                    break
            except queue.Empty:
                # 接続を維持するためにハートビートを送る
                yield 'data: {"type": "heartbeat"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/train/status")
async def train_status() -> dict[str, Any]:
    """訓練の現在状態を返す。

    ブラウザリロード後に訓練が進行中かどうかを確認するために使用する。
    running=True の場合、/api/train/stream に再接続して進捗を再受信できる。
    """
    return {
        "running": _train_state.get("running", False),
        "game_type": _train_state.get("game_type"),
    }


@app.post("/api/train/stop")
async def train_stop() -> dict[str, str]:
    """訓練を停止する。

    stop_event をセットしてバックグラウンドスレッドに停止を伝える。
    実際の停止はフェーズ間のチェックポイントで行われる。
    """
    if not _train_state.get("running"):
        raise HTTPException(400, "No training session running")
    _train_state["stop_event"].set()
    return {"status": "stopping"}


@app.post("/api/train/load")
async def train_load(req: NewGameRequest) -> dict[str, Any]:
    """訓練済みモデルを MCTS AI に読み込む。

    このエンドポイントを呼んだ後、/api/new-game で ai_type="mcts" を
    指定して対局を開始すると、訓練済みモデルが使用される。
    """
    model_path = f"best_model_{req.game_type}.pt"
    if not Path(model_path).exists():
        raise HTTPException(404, f"No trained model found: {model_path}")
    return {"status": "loaded", "model_path": model_path, "game_type": req.game_type}


@app.post("/api/auto-move/{game_id}")
async def auto_move(game_id: str) -> dict[str, Any]:
    """自動対戦: 現在の手番プレイヤーのAIが1手指す。

    AI同士の観戦モードで使用する。/api/new-game で sente_type にAI種別を
    指定して開始したゲームに対してのみ有効。フロントエンドが一定間隔で
    呼び出すことで、AIどうしが交互に指し手を進める。

    Returns:
        state:        更新後の局面情報
        move:         指した手のエンコード値
        move_decoded: 指した手の文字列表現（例: "b2-b3"）
        moved_by:     手を指したプレイヤー（0=先手, 1=後手）
    """
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, "Game not found")

    state: GameState = game["state"]
    game_type: str = game["game_type"]

    if state.is_terminal:
        raise HTTPException(400, "Game is already over")

    # 現在の手番プレイヤーのAI関数を取得
    moved_by = state.current_player
    fn = game.get("sente_fn") if moved_by == 0 else game.get("gote_fn")
    if fn is None:
        raise HTTPException(400, "Current player is human — use /api/move instead")

    move = fn(state)
    state = state.apply_move(move)
    game["state"] = state

    decode_fn = animal_decode if game_type == "animal" else full_decode
    decoded = decode_fn(move)
    # dict を表示用文字列に変換（JS で [object Object] にならないよう）
    if decoded["type"] == "board":
        fr, fc = decoded["from"]
        tr, tc = decoded["to"]
        move_str = f"({fr},{fc})→({tr},{tc})"
    else:  # drop
        tr, tc = decoded["to"]
        move_str = f"打({tr},{tc})"
    return {
        "state": _state_to_dict(state, game_type),
        "move": move,
        "move_decoded": move_str,
        "moved_by": moved_by,
    }


def main() -> None:
    """Run the web server.

    `uv run shogi-web` または `python -m shogi_ai.web.app` で起動する。
    ブラウザで http://localhost:8000 にアクセスして対局できる。
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
