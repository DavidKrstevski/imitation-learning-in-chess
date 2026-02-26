<script>
  import { Chess } from "chess.js";
  import { tick } from "svelte";

  const lichessPieceBase = "https://lichess1.org/assets/piece/cburnett";

  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = [8, 7, 6, 5, 4, 3, 2, 1];
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

  function squareName(row, col) {
    return `${files[col]}${ranks[row]}`;
  }

  function refresh() {
    board = game.board();
    const history = game.history({ verbose: true });
    lastMove = history.length ? history[history.length - 1] : null;
    moveRows = movePairsFromSan(game.history());
    checkedKingSquare = findCheckedKingSquare();
  }

  function pieceImage(square) {
    if (!square) return "";
    const color = square.color === "w" ? "w" : "b";
    const type = square.type.toUpperCase();
    return `${lichessPieceBase}/${color}${type}.svg`;
  }

  async function tryMove(from, to) {
    const move = await applyMoveAnimated({ from, to, promotion: "q" });
    return Boolean(move);
  }

  async function askModelMove() {
    busy = true;
    error = "";
    try {
      const uciMoves = game.history({ verbose: true }).map((m) => `${m.from}${m.to}${m.promotion ? m.promotion : ""}`);
      const response = await fetch("/api/model-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ moves: uciMoves })
      });

      if (!response.ok) {
        const msg = await response.text();
        throw new Error(msg || "API error");
      }

      const data = await response.json();
      if (!data.game_over && data.model_move) {
        await applyMoveAnimated(data.model_move);
      }
    } catch (e) {
      error = `Backend Fehler: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function onPlayerMove(from, to) {
    if (busy || game.isGameOver()) return;
    if (!(await tryMove(from, to))) {
      error = `Illegaler Zug: ${from}${to}`;
      return;
    }
    if (game.isGameOver()) return;
    await askModelMove();
  }

  function clickSquare(sq) {
    if (busy || game.isGameOver()) return;
    const legalTarget = getLegalMoveTarget(sq);

    if (!selected) {
      const piece = game.get(sq);
      if (!piece) return;
      const turn = game.turn();
      if (piece.color !== turn) return;
      setSelection(sq);
      return;
    }

    if (selected === sq) {
      clearSelection();
      return;
    }

    if (legalTarget) {
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
    if (!from) return;
    onPlayerMove(from, targetSq);
  }

  function resetGame() {
    game.reset();
    refresh();
    clearSelection();
    error = "";
  }

  function undoMove() {
    if (busy) return;
    const undone = game.undo();
    if (!undone) return;
    refresh();
    clearSelection();
    error = "";
  }

  function gameResultText() {
    if (!game.isGameOver()) return "";

    if (game.isCheckmate()) {
      const winner = game.turn() === "w" ? "b" : "w";
      if (winner === playerColor) return "Du hast gewonnen (Schachmatt).";
      return "Du hast verloren (Schachmatt).";
    }

    if (game.isStalemate()) return "Remis (Patt).";
    if (game.isThreefoldRepetition()) return "Remis (dreifache Stellungswiederholung).";
    if (game.isInsufficientMaterial()) return "Remis (ungenügendes Material).";
    if (game.isDraw()) return "Remis.";

    return "Spiel beendet.";
  }

  function movePairsFromSan(history) {
    const pairs = [];
    for (let i = 0; i < history.length; i += 2) {
      pairs.push({
        number: Math.floor(i / 2) + 1,
        white: history[i] ?? "",
        black: history[i + 1] ?? ""
      });
    }
    return pairs;
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

  function getLegalMoveTarget(sq) {
    for (const move of legalMoves) {
      if (move.to === sq) return move;
    }
    return null;
  }

  function isCheckNow() {
    if (typeof game.isCheck === "function") return game.isCheck();
    if (typeof game.inCheck === "function") return game.inCheck();
    return false;
  }

  function findCheckedKingSquare() {
    if (!isCheckNow()) return null;
    const colorInCheck = game.turn();
    for (let r = 0; r < board.length; r += 1) {
      for (let c = 0; c < board[r].length; c += 1) {
        const piece = board[r][c];
        if (piece && piece.type === "k" && piece.color === colorInCheck) {
          return squareName(r, c);
        }
      }
    }
    return null;
  }

  function squareCoords(sq) {
    const file = sq[0];
    const rank = Number(sq[1]);
    return {
      col: files.indexOf(file),
      row: ranks.indexOf(rank)
    };
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function isMovingSourceSquare(sq) {
    return Boolean(movingPiece && movingPiece.from === sq);
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
    const parsedMove = probe.move(moveInput);
    if (!parsedMove) return null;

    await animateMove(parsedMove.from, parsedMove.to);
    const move = game.move(moveInput);
    if (!move) return null;

    refresh();
    clearSelection();
    error = "";
    return move;
  }

  refresh();
</script>

<main>
  <h1>Chess vs Model</h1>

  <div class="status">
    <span>Zug: {game.turn() === "w" ? "White" : "Black"}</span>
    {#if game.isGameOver()}
      <span class="result">{gameResultText()}</span>
    {/if}
    {#if busy}
      <span>Model denkt...</span>
    {/if}
  </div>

  <div class="game-layout">
    <div class="board">
      {#each board as row, r}
        {#each row as square, c}
          {@const sq = squareName(r, c)}
          {@const isLastFrom = lastMove && lastMove.from === sq}
          {@const isLastTo = lastMove && lastMove.to === sq}
          {@const legalTarget = getLegalMoveTarget(sq)}
          {@const isCheckedKing = checkedKingSquare === sq}
          <button
            class="square {(r + c) % 2 === 0 ? 'light' : 'dark'} {selected === sq ? 'selected' : ''} {isLastFrom ? 'last-from' : ''} {isLastTo ? 'last-to' : ''} {legalTarget && !legalTarget.isCapture ? 'legal-target' : ''} {legalTarget && legalTarget.isCapture ? 'legal-capture' : ''} {isCheckedKing ? 'checked-king' : ''}"
            on:click={() => clickSquare(sq)}
            on:dragover={(e) => e.preventDefault()}
            on:drop={(e) => onDrop(e, sq)}
          >
            {#if square && !isMovingSourceSquare(sq)}
              <img
                class="piece-image"
                src={pieceImage(square)}
                alt={`${square.color === "w" ? "White" : "Black"} ${square.type}`}
                draggable="true"
                on:dragstart={(e) => onDragStart(e, sq)}
              />
            {/if}
          </button>
        {/each}
      {/each}

      {#if movingPiece}
        <img
          class="moving-piece {movingPiece.active ? 'active' : ''}"
          src={movingPiece.src}
          alt="moving chess piece"
          style="--row: {movingPiece.row}; --col: {movingPiece.col}; --dx: {movingPiece.dx}; --dy: {movingPiece.dy};"
        />
      {/if}
    </div>

    <aside class="moves-panel">
      <h2>Letzte Züge</h2>
      <div class="moves-list">
        {#if moveRows.length === 0}
          <div class="move-row empty">Noch keine Züge</div>
        {:else}
          {#each moveRows as pair}
            <div class="move-row">
              <span class="move-no">{pair.number}.</span>
              <span class="move-white">{pair.white}</span>
              <span class="move-black">{pair.black}</span>
            </div>
          {/each}
        {/if}
      </div>
    </aside>
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
  main {
    font-family: "Noto Sans", "Segoe UI", sans-serif;
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

  .board {
    position: relative;
    display: grid;
    grid-template-columns: repeat(8, minmax(36px, 72px));
    width: fit-content;
    overflow: hidden;
    border: none;
    border-radius: 16px;
  }

  .game-layout {
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }

  .square {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    padding: 0;
  }

  .light { background: #f0d9b5; }
  .dark { background: #b58863; }
  .selected { background: #829769; }
  .square.last-from::after,
  .square.last-to::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .square.last-from::after { background: #AAA23A }
  .square.last-to::after { background: #AAA23A }

  .square.legal-target::before,
  .square.legal-capture::before {
    content: "";
    position: absolute;
    pointer-events: none;
    z-index: 2;
  }

  .square.legal-target::before {
    width: 28%;
    height: 28%;
    border-radius: 999px;
    background: #829769;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .square.legal-capture::before {
    inset: 10%;
    border-radius: 999px;
    border: 4px solid #829769;
  }

  .square.checked-king {
    box-shadow: inset 0 0 0 3px rgba(198, 40, 40, 0.95), inset 0 0 18px rgba(198, 40, 40, 0.7);
  }

  .piece-image {
    position: relative;
    z-index: 1;
    width: 86%;
    height: 86%;
    object-fit: contain;
    user-select: none;
    -webkit-user-drag: element;
  }

  .moving-piece {
    position: absolute;
    z-index: 3;
    width: 12.5%;
    height: 12.5%;
    left: calc(var(--col) * 12.5%);
    top: calc(var(--row) * 12.5%);
    padding: 0.7%;
    box-sizing: border-box;
    pointer-events: none;
    transform: translate(0, 0);
    transition: transform 180ms cubic-bezier(0.2, 0, 0.2, 1);
  }

  .moving-piece.active {
    transform: translate(calc(var(--dx) * 100%), calc(var(--dy) * 100%));
  }

  .controls {
    margin-top: 12px;
    display: flex;
    gap: 8px;
  }

  .controls button {
    padding: 8px 12px;
    border: 1px solid #333;
    background: #ffffff;
    cursor: pointer;
  }

  .controls button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .result { color: #0d47a1; }

  .error { color: #b00020; }

  .moves-panel {
    min-width: 220px;
    border: 1px solid #c8c8c8;
    padding: 10px;
    background: #ffffff;
  }

  .moves-panel h2 {
    margin: 0 0 8px 0;
    font-size: 15px;
    color: #6b6b6b;
  }

  .moves-list {
    max-height: 430px;
    overflow-y: auto;
    font-size: 14px;
  }

  .move-row {
    display: grid;
    grid-template-columns: 36px 1fr 1fr;
    gap: 8px;
    padding: 3px 0;
    border-radius: 4px;
  }

  .move-row:nth-child(odd) {
    background: #f7f7f7;
  }

  .move-row.empty {
    display: block;
    color: #666;
  }

  @media (max-width: 900px) {
    .game-layout {
      flex-direction: column;
    }

    .moves-panel {
      width: 100%;
      min-width: unset;
    }
  }
</style>
