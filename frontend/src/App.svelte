<script>
  import { Chess } from "chess.js";

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

  function squareName(row, col) {
    return `${files[col]}${ranks[row]}`;
  }

  function refresh() {
    board = game.board();
    const history = game.history({ verbose: true });
    lastMove = history.length ? history[history.length - 1] : null;
    moveRows = movePairsFromSan(game.history());
  }

  function pieceImage(square) {
    if (!square) return "";
    const color = square.color === "w" ? "w" : "b";
    const type = square.type.toUpperCase();
    return `${lichessPieceBase}/${color}${type}.svg`;
  }

  function tryMove(from, to) {
    const move = game.move({ from, to, promotion: "q" });
    if (!move) return false;
    refresh();
    selected = null;
    error = "";
    return true;
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
        game.move(data.model_move);
        refresh();
      }
    } catch (e) {
      error = `Backend Fehler: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function onPlayerMove(from, to) {
    if (busy || game.isGameOver()) return;
    if (!tryMove(from, to)) {
      error = `Illegaler Zug: ${from}${to}`;
      return;
    }
    if (game.isGameOver()) return;
    await askModelMove();
  }

  function clickSquare(sq) {
    if (busy || game.isGameOver()) return;

    if (!selected) {
      const piece = game.get(sq);
      if (!piece) return;
      const turn = game.turn();
      if (piece.color !== turn) return;
      selected = sq;
      return;
    }

    if (selected === sq) {
      selected = null;
      return;
    }

    const from = selected;
    selected = null;
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
    selected = null;
    error = "";
  }

  function undoMove() {
    if (busy) return;
    const undone = game.undo();
    if (!undone) return;
    refresh();
    selected = null;
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

  function movePairs() {
    const history = game.history();
    return movePairsFromSan(history);
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
          <button
            class="square {(r + c) % 2 === 0 ? 'light' : 'dark'} {selected === sq ? 'selected' : ''} {isLastFrom ? 'last-from' : ''} {isLastTo ? 'last-to' : ''}"
            on:click={() => clickSquare(sq)}
            on:dragover={(e) => e.preventDefault()}
            on:drop={(e) => onDrop(e, sq)}
          >
            {#if square}
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
    display: grid;
    grid-template-columns: repeat(8, minmax(36px, 72px));
    width: fit-content;
    border: 1px solid #7a6a55;
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
  .selected { outline: 3px solid #1976d2; z-index: 1; }
  .square.last-from::after,
  .square.last-to::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .square.last-from::after { background: #AAA23A }
  .square.last-to::after { background: #AAA23A }

  .piece-image {
    position: relative;
    z-index: 1;
    width: 86%;
    height: 86%;
    object-fit: contain;
    user-select: none;
    -webkit-user-drag: element;
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
