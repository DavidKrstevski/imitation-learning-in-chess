<script>
  import { pieceImage, squareName } from "../lib/chessHelpers";

  export let board = [];
  export let selected = null;
  export let lastMove = null;
  export let legalMoves = [];
  export let checkedKingSquare = null;
  export let movingPiece = null;
  export let onSquareClick = () => {};
  export let onSquareDrop = () => {};
  export let onPieceDragStart = () => {};

  function legalTarget(sq) {
    return legalMoves.find((move) => move.to === sq) ?? null;
  }

  function isMovingSourceSquare(sq) {
    return Boolean(movingPiece && movingPiece.from === sq);
  }
</script>

<div class="board">
  {#each board as row, r}
    {#each row as square, c}
      {@const sq = squareName(r, c)}
      {@const isLastFrom = lastMove && lastMove.from === sq}
      {@const isLastTo = lastMove && lastMove.to === sq}
      {@const target = legalTarget(sq)}
      {@const isCheckedKing = checkedKingSquare === sq}
      <button
        class="square {(r + c) % 2 === 0 ? 'light' : 'dark'} {selected === sq ? 'selected' : ''} {isLastFrom ? 'last-from' : ''} {isLastTo ? 'last-to' : ''} {target && !target.isCapture ? 'legal-target' : ''} {target && target.isCapture ? 'legal-capture' : ''} {isCheckedKing ? 'checked-king' : ''}"
        on:click={() => onSquareClick(sq)}
        on:dragover={(e) => e.preventDefault()}
        on:drop={(e) => onSquareDrop(e, sq)}
      >
        {#if square && !isMovingSourceSquare(sq)}
          <img
            class="piece-image"
            src={pieceImage(square)}
            alt={`${square.color === "w" ? "White" : "Black"} ${square.type}`}
            draggable="true"
            on:dragstart={(e) => onPieceDragStart(e, sq)}
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

<style>
  .board {
    position: relative;
    display: grid;
    grid-template-columns: repeat(8, minmax(36px, 72px));
    width: fit-content;
    overflow: hidden;
    border: none;
    border-radius: 16px;
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

  .square.last-from::after { background: #aaa23a; }
  .square.last-to::after { background: #aaa23a; }

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
</style>
