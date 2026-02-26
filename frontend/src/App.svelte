<script>
  import { Chess } from "chess.js";

  const lichessPieceBase = "https://lichess1.org/assets/piece/cburnett";

  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = [8, 7, 6, 5, 4, 3, 2, 1];

  const game = new Chess();

  let board = game.board();
  let selected = null;
  let dragging = null;
  let busy = false;
  let error = "";

  function squareName(row, col) {
    return `${files[col]}${ranks[row]}`;
  }

  function refresh() {
    board = game.board();
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
</script>

<main>
  <h1>Chess vs Model</h1>

  <div class="status">
    <span>Zug: {game.turn() === "w" ? "White" : "Black"}</span>
    {#if game.isGameOver()}
      <span>Game over</span>
    {/if}
    {#if busy}
      <span>Model denkt...</span>
    {/if}
  </div>

  <div class="board">
    {#each board as row, r}
      {#each row as square, c}
        {@const sq = squareName(r, c)}
        <button
          class="square {(r + c) % 2 === 0 ? 'light' : 'dark'} {selected === sq ? 'selected' : ''}"
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

  <div class="controls">
    <button on:click={resetGame}>Neues Spiel</button>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  <p class="moves">{game.history().join(" ")}</p>
</main>

<style>
  main {
    font-family: "Segoe UI", sans-serif;
    max-width: 700px;
    margin: 24px auto;
    padding: 0 16px;
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
    border: 1px solid #222;
  }

  .square {
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

  .piece-image {
    width: 86%;
    height: 86%;
    object-fit: contain;
    user-select: none;
    -webkit-user-drag: element;
  }

  .controls { margin-top: 12px; }

  .controls button {
    padding: 8px 12px;
    border: 1px solid #333;
    background: #fff;
    cursor: pointer;
  }

  .error { color: #b00020; }

  .moves {
    margin-top: 12px;
    font-family: Consolas, monospace;
    white-space: pre-wrap;
  }
</style>
