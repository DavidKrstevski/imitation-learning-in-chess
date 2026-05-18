<script>
  import { Chess } from "chess.js";
  import { onDestroy, onMount, tick } from "svelte";
  import ChessBoard from "./components/ChessBoard.svelte";
  import MovesPanel from "./components/MovesPanel.svelte";
  import {
    gameResultText,
    isCheckNow,
    movePairsFromSan,
    pieceImage,
    sleep,
    squareCoords,
    squareName
  } from "./lib/chessHelpers";

  const playerColor = "w";
  const game = new Chess();

  let board = game.board();
  let selected = null;
  let dragging = null;
  let busy = false;
  let error = "";
  let lastMove = null;
  let moveRows = [];
  let legalMoves = [];
  let movingPiece = null;
  let checkedKingSquare = null;

  let players = [];
  let selectedPlayerId = "";
  let trainingUsername = "";
  let activeJob = null;
  let trainingBusy = false;
  let trainingError = "";
  let pollTimer = null;

  $: selectedPlayer = players.find((player) => player.id === selectedPlayerId) ?? null;
  $: trainingPercent = Math.round(((activeJob?.progress ?? 0) || 0) * 100);
  $: hasActiveTraining = activeJob && ["queued", "running"].includes(activeJob.status);

  function refresh() {
    board = game.board();
    const history = game.history({ verbose: true });
    lastMove = history.at(-1) ?? null;
    moveRows = movePairsFromSan(game.history());
    checkedKingSquare = findCheckedKingSquare();
  }

  function setSelection(sq) {
    selected = sq;
    legalMoves = game.moves({ square: sq, verbose: true }).map((move) => ({
      to: move.to,
      isCapture: Boolean(move.captured)
    }));
  }

  function clearSelection() {
    selected = null;
    legalMoves = [];
  }

  function findLegalTarget(sq) {
    return legalMoves.find((move) => move.to === sq) ?? null;
  }

  function findCheckedKingSquare() {
    if (!isCheckNow(game)) return null;
    const colorInCheck = game.turn();
    for (let r = 0; r < board.length; r += 1) {
      for (let c = 0; c < board[r].length; c += 1) {
        const piece = board[r][c];
        if (piece && piece.type === "k" && piece.color === colorInCheck) return squareName(r, c);
      }
    }
    return null;
  }

  function formatPercent(value) {
    if (typeof value !== "number") return "-";
    return `${(value * 100).toFixed(1)}%`;
  }

  async function apiJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = await response.text();
      try {
        const payload = JSON.parse(message);
        message = payload.detail || message;
      } catch {
        // Keep plain text response.
      }
      throw new Error(message || "API error");
    }
    return response.json();
  }

  async function loadPlayers() {
    const data = await apiJson("/api/players");
    players = data.players ?? [];
    if (!selectedPlayerId && players.length > 0) selectedPlayerId = players[0].id;
  }

  async function startTraining() {
    const username = trainingUsername.trim();
    if (!username || trainingBusy || hasActiveTraining) return;
    trainingBusy = true;
    trainingError = "";
    try {
      activeJob = await apiJson("/api/train-player", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username })
      });
      startPolling();
    } catch (e) {
      trainingError = e.message;
    } finally {
      trainingBusy = false;
    }
  }

  async function pollTrainingJob() {
    if (!activeJob?.id) return;
    try {
      activeJob = await apiJson(`/api/train-player/${activeJob.id}`);
      if (activeJob.status === "completed") {
        await loadPlayers();
        if (activeJob.player_id) selectedPlayerId = activeJob.player_id;
        stopPolling();
      } else if (activeJob.status === "failed") {
        stopPolling();
      }
    } catch (e) {
      trainingError = e.message;
      stopPolling();
    }
  }

  function startPolling() {
    stopPolling();
    pollTrainingJob();
    pollTimer = window.setInterval(pollTrainingJob, 2500);
  }

  function stopPolling() {
    if (pollTimer) window.clearInterval(pollTimer);
    pollTimer = null;
  }

  async function animateMove(from, to) {
    const piece = game.get(from);
    if (!piece) return;
    const fromPos = squareCoords(from);
    const toPos = squareCoords(to);
    if (fromPos.row < 0 || fromPos.col < 0 || toPos.row < 0 || toPos.col < 0) return;

    movingPiece = {
      from,
      src: pieceImage(piece),
      row: fromPos.row,
      col: fromPos.col,
      dx: toPos.col - fromPos.col,
      dy: toPos.row - fromPos.row,
      active: false
    };

    await tick();
    requestAnimationFrame(() => {
      if (movingPiece) movingPiece = { ...movingPiece, active: true };
    });
    await sleep(180);
    movingPiece = null;
  }

  async function applyMoveAnimated(moveInput) {
    const probe = new Chess(game.fen());
    const parsed = probe.move(moveInput);
    if (!parsed) return null;
    await animateMove(parsed.from, parsed.to);

    const move = game.move(moveInput);
    if (!move) return null;
    refresh();
    clearSelection();
    error = "";
    return move;
  }

  async function askModelMove() {
    busy = true;
    error = "";
    try {
      const uciMoves = game.history({ verbose: true }).map((m) => `${m.from}${m.to}${m.promotion ?? ""}`);
      const data = await apiJson("/api/model-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ moves: uciMoves, player_id: selectedPlayerId || null })
      });
      if (!data.game_over && data.model_move) await applyMoveAnimated(data.model_move);
    } catch (e) {
      error = `Backend Fehler: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function onPlayerMove(from, to) {
    if (busy || game.isGameOver()) return;
    if (!selectedPlayerId) {
      error = "Bitte zuerst links einen trainierten Spieler auswaehlen.";
      return;
    }
    if (!(await applyMoveAnimated({ from, to, promotion: "q" }))) {
      error = `Illegaler Zug: ${from}${to}`;
      return;
    }
    if (!game.isGameOver()) await askModelMove();
  }

  function clickSquare(sq) {
    if (busy || game.isGameOver()) return;
    if (!selected) {
      const piece = game.get(sq);
      if (piece && piece.color === game.turn()) setSelection(sq);
      return;
    }

    if (selected === sq) {
      clearSelection();
      return;
    }

    if (findLegalTarget(sq)) {
      const from = selected;
      clearSelection();
      onPlayerMove(from, sq);
      return;
    }

    const piece = game.get(sq);
    if (piece && piece.color === game.turn()) {
      setSelection(sq);
      return;
    }

    const from = selected;
    clearSelection();
    onPlayerMove(from, sq);
  }

  function onDragStart(event, sq) {
    if (busy || game.isGameOver()) return;
    const piece = game.get(sq);
    if (!piece || piece.color !== game.turn()) return;
    dragging = sq;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", sq);
  }

  function onDrop(event, targetSq) {
    event.preventDefault();
    const from = event.dataTransfer.getData("text/plain") || dragging;
    dragging = null;
    if (from) onPlayerMove(from, targetSq);
  }

  function undoMove() {
    if (busy || !game.undo()) return;
    refresh();
    clearSelection();
    error = "";
  }

  function resetGame() {
    game.reset();
    refresh();
    clearSelection();
    error = "";
  }

  function selectPlayer(playerId) {
    selectedPlayerId = playerId;
    resetGame();
  }

  onMount(async () => {
    refresh();
    try {
      await loadPlayers();
    } catch (e) {
      trainingError = e.message;
    }
  });

  onDestroy(stopPolling);

  refresh();
</script>

<main>
  <aside class="sidebar">
    <section class="panel">
      <h1>Spielertraining</h1>
      <form class="train-form" on:submit|preventDefault={startTraining}>
        <input
          bind:value={trainingUsername}
          placeholder="Lichess Username"
          autocomplete="off"
          disabled={trainingBusy || hasActiveTraining}
        />
        <button type="submit" disabled={!trainingUsername.trim() || trainingBusy || hasActiveTraining}>
          Start
        </button>
      </form>

      {#if activeJob}
        <div class="job">
          <div class="job-head">
            <strong>{activeJob.username}</strong>
            <span>{activeJob.status}</span>
          </div>
          <div class="progress-track">
            <div class="progress-bar" style="width: {trainingPercent}%;"></div>
          </div>
          <div class="job-meta">{activeJob.stage} - {trainingPercent}%</div>
          {#if activeJob.metrics && activeJob.status === "completed"}
            <div class="metrics">
              <span>Elo {activeJob.metrics.elo ?? "-"}</span>
              <span>Top1 {formatPercent(activeJob.metrics.finetuned_top1_accuracy)}</span>
            </div>
          {/if}
          {#if activeJob.error}
            <p class="error">{activeJob.error}</p>
          {/if}
        </div>
      {/if}

      {#if trainingError}
        <p class="error">{trainingError}</p>
      {/if}
    </section>

    <section class="panel">
      <h2>Trainierte Spieler</h2>
      <div class="player-list">
        {#if players.length === 0}
          <div class="empty">Noch kein Spieler trainiert.</div>
        {:else}
          {#each players as player}
            <button
              class:selected-player={selectedPlayerId === player.id}
              class="player-row"
              on:click={() => selectPlayer(player.id)}
            >
              <span class="player-name">{player.username}</span>
              <span class="player-score">
                Top1 {formatPercent(player.metrics?.finetuned_top1_accuracy)}
              </span>
            </button>
          {/each}
        {/if}
      </div>
    </section>
  </aside>

  <section class="play-area">
    <div class="topbar">
      <div>
        <div class="title">Chess vs Player Model</div>
        <div class="subtitle">
          {#if selectedPlayer}
            Gegner: {selectedPlayer.username}
          {:else}
            Kein Spieler ausgewaehlt
          {/if}
        </div>
      </div>
      <div class="status">
        <span>Zug: {game.turn() === "w" ? "White" : "Black"}</span>
        {#if game.isGameOver()}
          <span class="result">{gameResultText(game, playerColor)}</span>
        {/if}
        {#if busy}
          <span>Model denkt...</span>
        {/if}
      </div>
    </div>

    {#if selectedPlayer?.metrics}
      <div class="selected-metrics">
        <span>Train Games: {selectedPlayer.metrics.train_games_used ?? "-"}</span>
        <span>Elo: {selectedPlayer.metrics.elo ?? "-"}</span>
        <span>Top1: {formatPercent(selectedPlayer.metrics.finetuned_top1_accuracy)}</span>
      </div>
    {/if}

    <div class="game-layout">
      <ChessBoard
        {board}
        {selected}
        {lastMove}
        {legalMoves}
        {checkedKingSquare}
        {movingPiece}
        onSquareClick={clickSquare}
        onSquareDrop={onDrop}
        onPieceDragStart={onDragStart}
      />
      <MovesPanel {moveRows} />
    </div>

    <div class="controls">
      <button on:click={resetGame}>Neues Spiel</button>
      <button on:click={undoMove} disabled={busy || game.history().length === 0}>Zug zurück</button>
    </div>

    {#if error}
      <p class="error">{error}</p>
    {/if}
  </section>
</main>

<style>
  @import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap");

  main {
    font-family: "Manrope", "Segoe UI", sans-serif;
    min-height: 100vh;
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    gap: 24px;
    padding: 24px;
    color: #242424;
    background: #f4f6f5;
    box-sizing: border-box;
  }

  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .panel {
    background: #fff;
    border: 1px solid #d8dedb;
    border-radius: 8px;
    padding: 14px;
  }

  h1,
  h2 {
    margin: 0 0 12px 0;
    font-size: 18px;
  }

  h2 {
    font-size: 15px;
    color: #4f5d58;
  }

  .train-form {
    display: grid;
    grid-template-columns: 1fr 74px;
    gap: 8px;
  }

  input {
    min-width: 0;
    border: 1px solid #b9c3bf;
    border-radius: 6px;
    padding: 9px 10px;
    font: inherit;
  }

  button {
    border: 1px solid #26352f;
    border-radius: 6px;
    background: #fff;
    color: #1f2b27;
    cursor: pointer;
    font: inherit;
    font-weight: 600;
    padding: 8px 10px;
  }

  button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .job {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 13px;
  }

  .job-head,
  .metrics,
  .selected-metrics,
  .status {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    align-items: center;
  }

  .job-head {
    justify-content: space-between;
  }

  .progress-track {
    height: 8px;
    background: #e4e9e7;
    border-radius: 999px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: #2e7d57;
    transition: width 200ms ease;
  }

  .job-meta,
  .empty,
  .subtitle {
    color: #66736e;
    font-size: 13px;
  }

  .player-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 52vh;
    overflow-y: auto;
  }

  .player-row {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2px;
    text-align: left;
    border-color: #d8dedb;
  }

  .player-row.selected-player {
    background: #dfeee7;
    border-color: #2e7d57;
  }

  .player-name {
    font-weight: 700;
  }

  .player-score {
    color: #66736e;
    font-size: 12px;
  }

  .play-area {
    min-width: 0;
  }

  .topbar {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 12px;
  }

  .title {
    font-size: 24px;
    font-weight: 700;
  }

  .status {
    font-weight: 600;
    justify-content: flex-end;
  }

  .selected-metrics {
    margin-bottom: 12px;
    color: #4f5d58;
    font-size: 13px;
  }

  .game-layout {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 16px;
  }

  .controls {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .result {
    color: #0d47a1;
  }

  .error {
    color: #b00020;
    font-size: 13px;
  }

  @media (max-width: 980px) {
    main {
      grid-template-columns: 1fr;
      padding: 16px;
    }

    .game-layout,
    .topbar {
      flex-direction: column;
      align-items: center;
    }

    .topbar {
      align-items: flex-start;
    }
  }
</style>
