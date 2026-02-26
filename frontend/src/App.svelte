<script>
  import { Chess } from "chess.js";
  import { tick } from "svelte";
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
      const response = await fetch("/api/model-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ moves: uciMoves })
      });
      if (!response.ok) throw new Error((await response.text()) || "API error");

      const data = await response.json();
      if (!data.game_over && data.model_move) await applyMoveAnimated(data.model_move);
    } catch (e) {
      error = `Backend Fehler: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function onPlayerMove(from, to) {
    if (busy || game.isGameOver()) return;
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

  refresh();
</script>

<main>
  <h1>Chess vs Model</h1>

  <div class="status">
    <span>Zug: {game.turn() === "w" ? "White" : "Black"}</span>
    {#if game.isGameOver()}
      <span class="result">{gameResultText(game, playerColor)}</span>
    {/if}
    {#if busy}
      <span>Model denkt...</span>
    {/if}
  </div>

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
</main>

<style>
  @import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap");

  main {
    font-family: "Manrope", "Segoe UI", sans-serif;
    max-width: 980px;
    margin: 24px auto;
    padding: 0 16px;
    color: #2b2b2b;
  }

  h1 { margin-bottom: 12px; }

  .status {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    font-weight: 600;
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
    gap: 8px;
  }

  .controls button {
    padding: 8px 12px;
    border: 1px solid #333;
    background: #fff;
    cursor: pointer;
  }

  .controls button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .result { color: #0d47a1; }
  .error { color: #b00020; }

  @media (max-width: 900px) {
    .game-layout {
      flex-direction: column;
      align-items: center;
    }
  }
</style>
