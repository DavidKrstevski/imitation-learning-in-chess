export const lichessPieceBase = "https://lichess1.org/assets/piece/cburnett";
export const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
export const ranks = [8, 7, 6, 5, 4, 3, 2, 1];

export function squareName(row, col) {
  return `${files[col]}${ranks[row]}`;
}

export function pieceImage(square) {
  if (!square) return "";
  const color = square.color === "w" ? "w" : "b";
  const type = square.type.toUpperCase();
  return `${lichessPieceBase}/${color}${type}.svg`;
}

export function movePairsFromSan(history) {
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

export function squareCoords(sq) {
  const file = sq[0];
  const rank = Number(sq[1]);
  return {
    col: files.indexOf(file),
    row: ranks.indexOf(rank)
  };
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isCheckNow(game) {
  if (typeof game.isCheck === "function") return game.isCheck();
  if (typeof game.inCheck === "function") return game.inCheck();
  return false;
}

export function gameResultText(game, playerColor = "w") {
  if (!game.isGameOver()) return "";

  if (game.isCheckmate()) {
    const winner = game.turn() === "w" ? "b" : "w";
    return winner === playerColor
      ? "Du hast gewonnen (Schachmatt)."
      : "Du hast verloren (Schachmatt).";
  }

  if (game.isStalemate()) return "Remis (Patt).";
  if (game.isThreefoldRepetition()) return "Remis (dreifache Stellungswiederholung).";
  if (game.isInsufficientMaterial()) return "Remis (ungenügendes Material).";
  if (game.isDraw()) return "Remis.";
  return "Spiel beendet.";
}
