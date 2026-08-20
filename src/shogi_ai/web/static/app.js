// Shogi AI Web UI

const ANIMAL_PIECES = {
    0: '\u{1F424}', // CHICK
    1: '\u{1F992}', // GIRAFFE
    2: '\u{1F418}', // ELEPHANT
    3: '\u{1F981}', // LION
    4: '\u{1F414}', // HEN
};

const FULL_PIECES = {
    0: '\u6B69', 1: '\u9999', 2: '\u6842', 3: '\u9280', 4: '\u91D1',
    5: '\u89D2', 6: '\u98DB', 7: '\u7389',
    8: '\u3068', 9: '\u674F', 10: '\u572D', 11: '\u5168',
    12: '\u99AC', 13: '\u9F8D',
};

const ANIMAL_PIECE_VALUES = {
    'CHICK': 0, 'GIRAFFE': 1, 'ELEPHANT': 2, 'LION': 3, 'HEN': 4,
};

const FULL_PIECE_VALUES = {
    'PAWN': 0, 'LANCE': 1, 'KNIGHT': 2, 'SILVER': 3, 'GOLD': 4,
    'BISHOP': 5, 'ROOK': 6, 'KING': 7,
    'PRO_PAWN': 8, 'PRO_LANCE': 9, 'PRO_KNIGHT': 10, 'PRO_SILVER': 11,
    'HORSE': 12, 'DRAGON': 13,
};

const ANIMAL_HAND_NAMES = ['CHICK', 'GIRAFFE', 'ELEPHANT'];
const FULL_HAND_NAMES = ['PAWN', 'LANCE', 'KNIGHT', 'SILVER', 'GOLD', 'BISHOP', 'ROOK'];

let gameId = null;
let gameType = 'animal';
let gameState = null;
let selectedSquare = null;
let selectedHandPiece = null;
let lastAiFrom = -1;
let lastAiTo = -1;

// 観戦モード（AI vs AI）
let isSenteAI = false;
let autoPlayTimer = null;
let autoPlaySpeed = 600;  // ms/手
let isAutoPlayRunning = false;

async function newGame() {
    // 進行中の観戦を停止してから新規開始
    stopAutoPlay();

    gameType = document.getElementById('game-type').value;
    const aiType = document.getElementById('ai-type').value;
    const senteType = document.getElementById('sente-type').value;
    isSenteAI = senteType !== 'human';

    const res = await fetch('/api/new-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_type: gameType, ai_type: aiType, sente_type: senteType }),
    });
    const data = await res.json();
    gameId = data.game_id;
    gameState = data.state;
    selectedSquare = null;
    selectedHandPiece = null;
    lastAiFrom = -1;
    lastAiTo = -1;
    document.getElementById('move-list').innerHTML = '';

    const autoControls = document.getElementById('auto-play-controls');
    if (isSenteAI) {
        // AI vs AI: 観戦モードを表示して自動再生を開始
        autoControls.style.display = '';
        render();
        startAutoPlay();
    } else {
        autoControls.style.display = 'none';
        render();
    }

    updateRulesPanel(gameType);
}

function updateRulesPanel(type) {
    const animalRules = document.getElementById('rules-animal');
    const fullRules = document.getElementById('rules-full');
    if (type === 'animal') {
        animalRules.style.display = '';
        fullRules.style.display = 'none';
    } else {
        animalRules.style.display = 'none';
        fullRules.style.display = '';
    }
}

async function makeMove(move) {
    if (!gameId || gameState.is_terminal) return;
    selectedSquare = null;
    selectedHandPiece = null;

    const res = await fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId, move: move }),
    });

    if (!res.ok) {
        const err = await res.json();
        alert(err.detail);
        return;
    }

    const data = await res.json();
    gameState = data.state;

    // Track AI's last move for highlighting
    if (data.ai_move !== null) {
        const ns = gameState.rows * gameState.cols;
        lastAiFrom = getFromIdx(data.ai_move, ns);
        lastAiTo = getToIdx(data.ai_move, ns);
    } else {
        lastAiFrom = -1;
        lastAiTo = -1;
    }

    const moveList = document.getElementById('move-list');
    moveList.innerHTML += `<div>You: move ${data.player_move}</div>`;
    if (data.ai_move !== null) {
        moveList.innerHTML += `<div>AI: move ${data.ai_move}</div>`;
    }
    moveList.scrollTop = moveList.scrollHeight;

    render();
}

// --- Move decoding helpers ---

function getFromIdx(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return Math.floor(move / 12);
        return -1; // drop
    } else {
        if (move < numSquares * numSquares) return Math.floor(move / numSquares);
        if (move < 2 * numSquares * numSquares)
            return Math.floor((move - numSquares * numSquares) / numSquares);
        return -1; // drop
    }
}

function getToIdx(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return move % 12;
        return (move - 144) % 12;
    } else {
        if (move < numSquares * numSquares) return move % numSquares;
        if (move < 2 * numSquares * numSquares)
            return (move - numSquares * numSquares) % numSquares;
        return (move - 2 * numSquares * numSquares) % numSquares;
    }
}

function isDropMove(move, numSquares) {
    if (gameType === 'animal') return move >= 144;
    return move >= 2 * numSquares * numSquares;
}

function getDropPieceIndex(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return -1;
        return Math.floor((move - 144) / 12);
    } else {
        const dropBase = 2 * numSquares * numSquares;
        if (move < dropBase) return -1;
        return Math.floor((move - dropBase) / numSquares);
    }
}

// --- Rendering ---

function render() {
    if (!gameState) return;

    const pieces = gameType === 'animal' ? ANIMAL_PIECES : FULL_PIECES;
    const boardEl = document.getElementById('board');
    const { rows, cols, squares, hands, legal_moves, current_player, is_terminal, winner } = gameState;
    const numSquares = rows * cols;

    boardEl.className = `board ${gameType}`;
    boardEl.innerHTML = '';

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = document.createElement('div');
            cell.className = 'cell';

            const sq = squares[idx];
            if (sq) {
                const pieceEl = document.createElement('span');
                const isPromoted = sq.type >= 8; // PRO_PAWN=8..DRAGON=13
                pieceEl.className = `piece ${sq.owner === 1 ? 'gote' : 'sente'}${isPromoted && gameType === 'full' ? ' promoted' : ''}`;
                pieceEl.textContent = pieces[sq.type] || '?';
                cell.appendChild(pieceEl);
            }

            // Highlight AI's last move
            if (lastAiFrom === idx) cell.classList.add('last-move-from');
            if (lastAiTo === idx) cell.classList.add('last-move-to');

            // Highlight selected square
            if (selectedSquare === idx) {
                cell.classList.add('selected');
            }

            // Highlight legal targets for selected board piece
            if (selectedSquare !== null && !is_terminal && current_player === 0) {
                const isTarget = legal_moves.some(m =>
                    getFromIdx(m, numSquares) === selectedSquare &&
                    getToIdx(m, numSquares) === idx
                );
                if (isTarget) {
                    cell.classList.add('legal-target');
                }
            }

            // Highlight legal targets for selected hand piece (drop)
            if (selectedHandPiece !== null && !is_terminal && current_player === 0) {
                const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
                const ptIndex = handNames.indexOf(selectedHandPiece);
                const isTarget = legal_moves.some(m =>
                    isDropMove(m, numSquares) &&
                    getDropPieceIndex(m, numSquares) === ptIndex &&
                    getToIdx(m, numSquares) === idx
                );
                if (isTarget) {
                    cell.classList.add('legal-target');
                }
            }

            cell.addEventListener('click', () => handleCellClick(idx));
            boardEl.appendChild(cell);
        }
    }

    // Hands with active-turn highlight
    const goteHandEl = document.getElementById('gote-hand');
    const senteHandEl = document.getElementById('sente-hand');
    goteHandEl.className = `hand${current_player === 1 && !is_terminal ? ' active-turn' : ''}`;
    senteHandEl.className = `hand${current_player === 0 && !is_terminal ? ' active-turn' : ''}`;
    renderHand('gote-hand', hands[1], pieces, 1);
    renderHand('sente-hand', hands[0], pieces, 0);

    // Status
    const statusEl = document.getElementById('status');
    if (is_terminal) {
        if (winner === 0) {
            statusEl.textContent = isSenteAI ? '先手AIの勝利！' : 'You win!';
            statusEl.className = 'status win';
        } else if (winner === 1) {
            statusEl.textContent = isSenteAI ? '後手AIの勝利！' : 'AIの勝ち！';
            statusEl.className = 'status lose';
        } else {
            statusEl.textContent = '引き分け！';
            statusEl.className = 'status';
        }
    } else if (isSenteAI) {
        // 観戦モード: 先手/後手どちらのAIが考えているか表示
        const turnName = current_player === 0 ? '先手AI' : '後手AI';
        statusEl.textContent = `${turnName} が考えています...`;
        statusEl.className = 'status ai-turn';
    } else {
        if (current_player === 0) {
            statusEl.textContent = 'あなたの番です（先手）';
            statusEl.className = 'status your-turn';
        } else {
            statusEl.textContent = 'AIが考えています...';
            statusEl.className = 'status ai-turn';
        }
    }

    // Hide legal moves buttons (kept in DOM for debugging)
    document.getElementById('legal-moves').style.display = 'none';
}

// --- Click handlers ---

function handleCellClick(idx) {
    // 観戦モード（AI vs AI）またはゲームが終了している場合は操作不可
    if (!gameState || gameState.is_terminal || isSenteAI) return;
    if (gameState.current_player !== 0) return;

    const numSquares = gameState.rows * gameState.cols;
    const { squares, legal_moves } = gameState;

    // Board piece is selected
    if (selectedSquare !== null) {
        // Click same square -> deselect
        if (selectedSquare === idx) {
            selectedSquare = null;
            render();
            return;
        }

        // Try to move to this cell
        const matching = legal_moves.filter(m =>
            getFromIdx(m, numSquares) === selectedSquare &&
            getToIdx(m, numSquares) === idx
        );
        if (matching.length === 1) {
            makeMove(matching[0]);
            return;
        } else if (matching.length >= 2) {
            // Promotion choice (lower index = no promote, higher = promote)
            const choice = confirm('\u6210\u308A\u307E\u3059\u304B\uFF1F (OK=\u6210\u308A, Cancel=\u4E0D\u6210)');
            matching.sort((a, b) => a - b);
            makeMove(choice ? matching[matching.length - 1] : matching[0]);
            return;
        }

        // Not a legal target — switch to another own piece if possible
        const sq = squares[idx];
        if (sq && sq.owner === 0) {
            const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
            if (hasLegalMoves) {
                selectedSquare = idx;
                selectedHandPiece = null;
                render();
                return;
            }
        }

        // Deselect
        selectedSquare = null;
        selectedHandPiece = null;
        render();
        return;
    }

    // Hand piece is selected — try to drop on this cell
    if (selectedHandPiece !== null) {
        const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
        const ptIndex = handNames.indexOf(selectedHandPiece);
        const matching = legal_moves.filter(m =>
            isDropMove(m, numSquares) &&
            getDropPieceIndex(m, numSquares) === ptIndex &&
            getToIdx(m, numSquares) === idx
        );
        if (matching.length === 1) {
            makeMove(matching[0]);
            return;
        }

        // Not a valid drop target — switch to board piece if possible
        const sq = squares[idx];
        if (sq && sq.owner === 0) {
            const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
            if (hasLegalMoves) {
                selectedSquare = idx;
                selectedHandPiece = null;
                render();
                return;
            }
        }

        // Deselect
        selectedSquare = null;
        selectedHandPiece = null;
        render();
        return;
    }

    // Nothing selected — select own piece
    const sq = squares[idx];
    if (sq && sq.owner === 0) {
        const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
        if (hasLegalMoves) {
            selectedSquare = idx;
            selectedHandPiece = null;
            render();
        }
    }
}

function handleHandPieceClick(pieceName) {
    if (!gameState || gameState.is_terminal || gameState.current_player !== 0) return;

    const numSquares = gameState.rows * gameState.cols;
    const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
    const ptIndex = handNames.indexOf(pieceName);

    const hasDropMoves = gameState.legal_moves.some(m =>
        isDropMove(m, numSquares) && getDropPieceIndex(m, numSquares) === ptIndex
    );

    if (hasDropMoves) {
        if (selectedHandPiece === pieceName) {
            selectedHandPiece = null; // toggle off
        } else {
            selectedHandPiece = pieceName;
        }
        selectedSquare = null;
        render();
    }
}

function renderHand(elementId, handPieces, pieceMap, owner) {
    const el = document.getElementById(elementId);
    const piecesEl = el.querySelector('.hand-pieces');
    piecesEl.innerHTML = '';

    if (!handPieces || handPieces.length === 0) {
        piecesEl.textContent = '-';
        return;
    }

    const values = gameType === 'animal' ? ANIMAL_PIECE_VALUES : FULL_PIECE_VALUES;

    for (const name of handPieces) {
        const span = document.createElement('span');
        span.className = 'hand-piece';
        if (owner === 0 && selectedHandPiece === name) {
            span.classList.add('selected');
        }
        span.textContent = pieceMap[values[name]] || name;
        if (owner === 0) {
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                handleHandPieceClick(name);
            });
        }
        piecesEl.appendChild(span);
    }
}

// ─── 観戦モード（自動対戦） ───────────────────────────────────────────────────

async function autoPlayStep() {
    if (!gameId || !isAutoPlayRunning || !gameState || gameState.is_terminal) return;

    const res = await fetch(`/api/auto-move/${gameId}`, { method: 'POST' });
    if (!res.ok) {
        stopAutoPlay();
        return;
    }

    const data = await res.json();
    gameState = data.state;

    // 最後に指した手をハイライト
    const ns = gameState.rows * gameState.cols;
    lastAiFrom = getFromIdx(data.move, ns);
    lastAiTo = getToIdx(data.move, ns);

    const moveList = document.getElementById('move-list');
    const playerName = data.moved_by === 0 ? '先手' : '後手';
    moveList.innerHTML += `<div>${playerName}: ${data.move_decoded}</div>`;
    moveList.scrollTop = moveList.scrollHeight;

    render();

    if (!gameState.is_terminal && isAutoPlayRunning) {
        autoPlayTimer = setTimeout(autoPlayStep, autoPlaySpeed);
    } else {
        isAutoPlayRunning = false;
        updateAutoPlayButton();
    }
}

function startAutoPlay() {
    if (isAutoPlayRunning) return;
    isAutoPlayRunning = true;
    updateAutoPlayButton();
    autoPlayStep();
}

function stopAutoPlay() {
    isAutoPlayRunning = false;
    if (autoPlayTimer !== null) {
        clearTimeout(autoPlayTimer);
        autoPlayTimer = null;
    }
    updateAutoPlayButton();
}

function toggleAutoPlay() {
    if (isAutoPlayRunning) {
        stopAutoPlay();
    } else {
        startAutoPlay();
    }
}

function updateAutoPlayButton() {
    const btn = document.getElementById('auto-play-btn');
    if (!btn) return;
    btn.textContent = isAutoPlayRunning ? '⏸ 一時停止' : '▶ 再生';
}

function setAutoPlaySpeed() {
    autoPlaySpeed = parseInt(document.getElementById('auto-play-speed').value, 10);
}

// ─── タブ管理 ────────────────────────────────────────────────────────────────

function showTab(name) {
    document.getElementById('tab-game').style.display = name === 'game' ? '' : 'none';
    document.getElementById('tab-training').style.display = name === 'training' ? '' : 'none';
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab')[name === 'game' ? 0 : 1].classList.add('active');
}

// ─── 訓練 UI ─────────────────────────────────────────────────────────────────

let trainEventSource = null;
let trainGameType = 'animal';

function onTrainGameTypeChange() {
    const gameType = document.getElementById('train-game-type').value;
    // ゲーム種別に応じたデフォルト世代数を設定
    document.getElementById('train-generations').value = gameType === 'animal' ? 3 : 1;
}

function addTrainLog(msg) {
    const log = document.getElementById('train-log');
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function updatePhaseLabel(phase) {
    const labels = {
        self_play: '⏳ 自己対局中...',
        training:  '🧠 ネットワーク訓練中...',
        arena:     '⚔️  アリーナ対戦中...',
    };
    document.getElementById('train-phase-label').textContent = labels[phase] || phase;
}

function updateProgress(gen, total) {
    const pct = total > 0 ? Math.round((gen / total) * 100) : 0;
    document.getElementById('progress-bar').style.width = `${pct}%`;
    document.getElementById('train-gen-label').textContent = `${gen} / ${total} 世代`;
}

async function startTraining() {
    trainGameType = document.getElementById('train-game-type').value;
    const numGen = parseInt(document.getElementById('train-generations').value, 10);

    // 本将棋 × 多世代は非常に時間がかかるため確認ダイアログを表示
    if (trainGameType === 'full' && numGen > 2) {
        const hours = numGen * 4;  // 1世代あたり目安4時間
        const ok = confirm(
            `本将棋 × ${numGen}世代の訓練は非常に時間がかかります。\n\n` +
            `目安: 合計 ${hours} 時間以上（環境により大きく異なります）\n\n` +
            `途中で止める場合は「停止」ボタンを押してください。\n` +
            `ブラウザを閉じてもサーバーが動いている限り訓練は続きます。\n\n` +
            `続けますか？`
        );
        if (!ok) return;
    }

    document.getElementById('train-start-btn').disabled = true;
    document.getElementById('train-stop-btn').disabled = false;
    document.getElementById('train-status-bar').style.display = '';
    document.getElementById('train-results-area').style.display = '';
    document.getElementById('train-done-area').style.display = 'none';
    document.getElementById('train-table-body').innerHTML = '';
    document.getElementById('train-log').innerHTML = '';
    updateProgress(0, numGen);

    const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_type: trainGameType, num_generations: numGen }),
    });
    if (!res.ok) {
        const err = await res.json();
        alert(err.detail);
        document.getElementById('train-start-btn').disabled = false;
        document.getElementById('train-stop-btn').disabled = true;
        return;
    }

    addTrainLog(`訓練開始: ${trainGameType}, ${numGen}世代`);
    listenTrainStream();
}

// SSE ストリームへの接続（startTraining と リロード後再接続 の両方から呼ぶ）
function listenTrainStream() {
    if (trainEventSource) trainEventSource.close();
    trainEventSource = new EventSource('/api/train/stream');
    trainEventSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleTrainingEvent(event);
    };
    trainEventSource.onerror = () => {
        addTrainLog('接続が切れました');
        trainEventSource.close();
        document.getElementById('train-start-btn').disabled = false;
        document.getElementById('train-stop-btn').disabled = true;
    };
}

function handleTrainingEvent(event) {
    if (event.type === 'heartbeat') return;

    if (event.type === 'phase') {
        updatePhaseLabel(event.phase);
        if (event.phase === 'self_play') {
            updateProgress(event.generation - 1, event.total);
            addTrainLog(`第${event.generation}世代 開始`);
        }
        if (event.phase === 'training' && event.data_size) {
            addTrainLog(`  └ 自己対局完了: ${event.data_size}局面`);
        }
    }

    if (event.type === 'generation_done') {
        updateProgress(event.generation, event.total);
        document.getElementById('train-phase-label').textContent = `✓ 第${event.generation}世代 完了`;

        // 世代結果を表に追加
        const tbody = document.getElementById('train-table-body');
        const tr = document.createElement('tr');
        const adoptedCell = event.adopted
            ? '<td class="adopted-yes">✓ 採用</td>'
            : '<td class="adopted-no">✗ 見送</td>';
        const winPct = Math.round(event.win_rate * 100);
        tr.innerHTML = `
            <td>${event.generation}</td>
            <td>${event.data_size}</td>
            <td>${event.policy_loss}</td>
            <td>${event.value_loss}</td>
            <td>${event.new_wins}</td>
            <td>${event.old_wins}</td>
            <td>${event.draws}</td>
            <td>${winPct}%</td>
            ${adoptedCell}
        `;
        tbody.appendChild(tr);

        const adoptMsg = event.adopted ? '✓ 採用' : '✗ 見送り';
        addTrainLog(
            `  └ 結果: 損失=${event.total_loss} 勝率=${winPct}% ${adoptMsg}`
        );
    }

    if (event.type === 'done') {
        document.getElementById('train-phase-label').textContent = '✅ 訓練完了！';
        document.getElementById('train-start-btn').disabled = false;
        document.getElementById('train-stop-btn').disabled = true;
        document.getElementById('train-done-area').style.display = '';
        document.getElementById('train-done-msg').textContent =
            `訓練完了！ モデルが保存されました。対局タブで MCTS AI と対戦できます。`;
        addTrainLog('訓練完了');
        if (trainEventSource) trainEventSource.close();
    }

    if (event.type === 'stopped') {
        document.getElementById('train-phase-label').textContent = '⏹ 停止しました';
        document.getElementById('train-start-btn').disabled = false;
        document.getElementById('train-stop-btn').disabled = true;
        addTrainLog('訓練を停止しました');
        if (trainEventSource) trainEventSource.close();
    }
}

async function stopTraining() {
    await fetch('/api/train/stop', { method: 'POST' });
    addTrainLog('停止リクエスト送信...');
}

async function loadTrainedModel() {
    const res = await fetch('/api/train/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_type: trainGameType }),
    });
    if (res.ok) {
        addTrainLog('訓練済みモデルを読み込みました');
        // 対局タブに切り替えてゲーム種別・AI種別を設定
        showTab('game');
        document.getElementById('game-type').value = trainGameType;
        document.getElementById('ai-type').value = 'mcts';
        newGame();
    } else {
        const err = await res.json();
        alert(err.detail);
    }
}

// Auto-start
document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('new-game-btn').addEventListener('click', newGame);
    // ゲーム種別変更時に即座に盤面を切り替える
    document.getElementById('game-type').addEventListener('change', newGame);
    newGame();

    // リロード後の復帰: 訓練が進行中なら訓練タブに戻ってSSEを再接続する
    try {
        const res = await fetch('/api/train/status');
        const data = await res.json();
        if (data.running) {
            showTab('training');
            if (data.game_type) trainGameType = data.game_type;
            document.getElementById('train-start-btn').disabled = true;
            document.getElementById('train-stop-btn').disabled = false;
            document.getElementById('train-status-bar').style.display = '';
            document.getElementById('train-phase-label').textContent = '訓練中（再接続）...';
            addTrainLog('ページを再読み込みしました。訓練は継続中です。停止するには「停止」ボタンを押してください。');
            listenTrainStream();
        }
    } catch (_) {
        // サーバー未起動などの場合は無視
    }
});
