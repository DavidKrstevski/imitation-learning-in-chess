from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import chess
import torch
import torch.nn.functional as F


DEFAULT_MAX_CONTEXT_MOVES = 24


def mirror_move_uci(uci: str) -> str:
    move = chess.Move.from_uci(uci)
    mirrored = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )
    return mirrored.uci()


def normalize_move_uci(uci: str, user_color: str) -> str:
    return mirror_move_uci(uci) if user_color == "black" else uci


def denormalize_move_uci(uci: str, user_color: str) -> str:
    return mirror_move_uci(uci) if user_color == "black" else uci


def normalize_board(board: chess.Board, user_color: str) -> chess.Board:
    return board.mirror() if user_color == "black" else board.copy()


def board_key(board: chess.Board) -> str:
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return f"{board.board_fen()} {'w' if board.turn else 'b'} {board.castling_xfen()} {ep}"


def infer_phase(board: chess.Board, ply_idx: int) -> str:
    non_pawn_non_king = sum(
        1
        for piece in board.piece_map().values()
        if piece.piece_type not in (chess.PAWN, chess.KING)
    )
    if ply_idx < 16:
        return "opening"
    if non_pawn_non_king <= 6:
        return "endgame"
    return "middlegame"


def split_games_game_level(
    game_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    test_frac: float = 0.2,
    val_frac_within_train: float = 0.2,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = [dict(row) for row in game_rows]
    rng = torch.Generator()
    rng.manual_seed(seed)

    if len(rows) < 5:
        raise ValueError("Need at least 5 games for a stable train/val/test split.")
    if not (0.0 < test_frac < 0.5):
        raise ValueError("test_frac must be between 0 and 0.5.")
    if not (0.0 < val_frac_within_train < 0.5):
        raise ValueError("val_frac_within_train must be between 0 and 0.5.")

    order = torch.randperm(len(rows), generator=rng).tolist()
    rows = [rows[idx] for idx in order]

    test_n = max(1, int(round(len(rows) * test_frac)))
    train_val_rows = rows[:-test_n]
    test_rows = rows[-test_n:]

    val_n = max(1, int(round(len(train_val_rows) * val_frac_within_train)))
    train_rows = train_val_rows[:-val_n]
    val_rows = train_val_rows[-val_n:]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Split produced an empty train, val, or test partition.")

    assert_no_overlap(
        {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        }
    )
    return train_rows, val_rows, test_rows


def assert_no_overlap(split_map: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    seen: dict[str, str] = {}
    for split_name, rows in split_map.items():
        for row in rows:
            game_id = str(row["id"])
            previous = seen.get(game_id)
            if previous is not None and previous != split_name:
                raise ValueError(f"Game id {game_id!r} appears in both {previous} and {split_name}.")
            seen[game_id] = split_name


def build_prompt(example: Mapping[str, object], *, max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES) -> str:
    norm_context_moves = list(example.get("norm_context_moves", []))
    tail_moves = norm_context_moves[-max_context_moves:] if max_context_moves > 0 else norm_context_moves
    moves_text = " ".join(tail_moves)
    return f"FEN: {example['norm_fen']}\nMOVES: {moves_text}\nMOVE:"


def build_position_examples(
    game_rows: Sequence[Mapping[str, object]],
    *,
    min_context_ply: int = 0,
    max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES,
) -> list[dict]:
    examples: list[dict] = []
    for game in game_rows:
        user_color = str(game["user_color"])
        board = chess.Board()
        orig_moves_so_far: list[str] = []
        norm_moves_so_far: list[str] = []

        for ply_idx, uci in enumerate(game["uci_moves"]):
            mover_matches_user = (
                (user_color == "white" and board.turn == chess.WHITE)
                or (user_color == "black" and board.turn == chess.BLACK)
            )

            if mover_matches_user and ply_idx >= min_context_ply:
                norm_board = normalize_board(board, user_color)
                legal_moves_norm = [normalize_move_uci(move.uci(), user_color) for move in board.legal_moves]
                context_tail = (
                    norm_moves_so_far[-max_context_moves:] if max_context_moves > 0 else list(norm_moves_so_far)
                )
                example = {
                    "game_id": str(game["id"]),
                    "side": user_color,
                    "opening_name": game.get("opening_name"),
                    "ply_idx": int(ply_idx),
                    "move_number": int(board.fullmove_number),
                    "phase": infer_phase(norm_board, ply_idx),
                    "orig_context_moves": list(orig_moves_so_far),
                    "orig_context_str": " ".join(orig_moves_so_far),
                    "orig_target": str(uci),
                    "norm_context_moves": list(norm_moves_so_far),
                    "norm_context_tail_moves": context_tail,
                    "norm_context_str": " ".join(norm_moves_so_far),
                    "norm_context_tail_str": " ".join(context_tail),
                    "norm_fen": norm_board.fen(),
                    "position_key": board_key(norm_board),
                    "norm_target": normalize_move_uci(str(uci), user_color),
                    "legal_moves_norm": legal_moves_norm,
                }
                example["prompt"] = build_prompt(example, max_context_moves=max_context_moves)
                examples.append(example)

            board.push(chess.Move.from_uci(str(uci)))
            orig_moves_so_far.append(str(uci))
            norm_moves_so_far.append(normalize_move_uci(str(uci), user_color))
    return examples


def build_scoring_batch(
    tokenizer,
    prompt_text: str,
    candidate_moves: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Prompt tokenization returned no ids.")

    seqs: list[list[int]] = []
    prompt_lens: list[int] = []
    for move in candidate_moves:
        move_ids = tokenizer(" " + move, add_special_tokens=False)["input_ids"]
        if not move_ids:
            move_ids = tokenizer(move, add_special_tokens=False)["input_ids"]
        if not move_ids:
            raise ValueError(f"Move tokenization returned no ids for {move!r}.")
        seqs.append(prompt_ids + move_ids)
        prompt_lens.append(len(prompt_ids))

    max_len = max(len(seq) for seq in seqs)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    input_ids = []
    attention_mask = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(prompt_lens, dtype=torch.long),
    )


def _score_candidate_batch(
    model,
    tokenizer,
    prompt_text: str,
    candidate_moves: Sequence[str],
    device: str,
) -> list[float]:
    if not candidate_moves:
        return []

    input_ids, attention_mask, prompt_lens = build_scoring_batch(tokenizer, prompt_text, candidate_moves)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lens = prompt_lens.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(shift_labels.shape[1], device=device).unsqueeze(0)
    candidate_mask = positions >= (prompt_lens.unsqueeze(1) - 1)
    valid_mask = shift_mask & candidate_mask

    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = token_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    scores = (gathered * valid_mask).sum(dim=1)
    return [float(score.item()) for score in scores]


def score_legal_moves(
    model,
    tokenizer,
    example: Mapping[str, object],
    device: str,
    *,
    batch_size: int = 64,
    max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES,
    candidate_moves: Sequence[str] | None = None,
) -> list[dict]:
    legal_moves = list(candidate_moves or example["legal_moves_norm"])
    if not legal_moves:
        return []

    prompt_text = str(example.get("prompt") or build_prompt(example, max_context_moves=max_context_moves))
    rows: list[dict] = []
    for start in range(0, len(legal_moves), batch_size):
        chunk = legal_moves[start : start + batch_size]
        chunk_scores = _score_candidate_batch(model, tokenizer, prompt_text, chunk, device)
        for move, score in zip(chunk, chunk_scores):
            rows.append(
                {
                    "norm_move": move,
                    "orig_move": denormalize_move_uci(move, str(example["side"])),
                    "score": score,
                }
            )

    rows.sort(key=lambda item: (-item["score"], item["norm_move"]))
    return rows


def score_candidate_moves(
    model,
    tokenizer,
    example: Mapping[str, object],
    candidate_moves: Sequence[str],
    device: str,
    *,
    max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES,
) -> list[dict]:
    prompt_text = str(example.get("prompt") or build_prompt(example, max_context_moves=max_context_moves))
    scores = _score_candidate_batch(model, tokenizer, prompt_text, candidate_moves, device)
    rows = []
    for move, score in zip(candidate_moves, scores):
        rows.append(
            {
                "norm_move": move,
                "orig_move": denormalize_move_uci(move, str(example["side"])),
                "score": score,
            }
        )
    return rows


def _finalize_bucket_stats(counter_map: Mapping[str, Counter]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for bucket_name, counts in counter_map.items():
        total = int(counts["total"])
        out[bucket_name] = {
            "top1": (counts["top1_correct"] / total) if total else 0.0,
            "top5": (counts["top5_correct"] / total) if total else 0.0,
            "total": total,
            "top1_correct": int(counts["top1_correct"]),
            "top5_correct": int(counts["top5_correct"]),
        }
    return out


def evaluate_ranker(
    model,
    tokenizer,
    eval_examples: Sequence[Mapping[str, object]],
    device: str,
    *,
    top_k: int = 5,
    debug_n: int = 10,
    max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES,
) -> dict:
    top1_correct = 0
    top5_correct = 0
    debug_rows: list[dict] = []
    by_side: defaultdict[str, Counter] = defaultdict(Counter)
    by_phase: defaultdict[str, Counter] = defaultdict(Counter)

    for example in eval_examples:
        ranked = score_legal_moves(
            model,
            tokenizer,
            example,
            device,
            max_context_moves=max_context_moves,
        )
        top_rows = ranked[:top_k]
        norm_preds = [row["norm_move"] for row in top_rows]
        orig_preds = [row["orig_move"] for row in top_rows]

        target_norm = str(example["norm_target"])
        target_orig = str(example["orig_target"])
        top1_hit = bool(norm_preds and norm_preds[0] == target_norm)
        top5_hit = target_norm in norm_preds

        if top1_hit:
            top1_correct += 1
        if top5_hit:
            top5_correct += 1

        side = str(example["side"])
        phase = str(example["phase"])
        by_side[side]["total"] += 1
        by_phase[phase]["total"] += 1
        if top1_hit:
            by_side[side]["top1_correct"] += 1
            by_phase[phase]["top1_correct"] += 1
        if top5_hit:
            by_side[side]["top5_correct"] += 1
            by_phase[phase]["top5_correct"] += 1

        if len(debug_rows) < debug_n:
            debug_rows.append(
                {
                    "game_id": example["game_id"],
                    "side": side,
                    "phase": phase,
                    "opening_name": example.get("opening_name"),
                    "target_norm": target_norm,
                    "target_orig": target_orig,
                    "top1_pred_norm": norm_preds[0] if norm_preds else None,
                    "top1_pred_orig": orig_preds[0] if orig_preds else None,
                    "top5_norm": norm_preds,
                    "top5_orig": orig_preds,
                    "top1_correct": top1_hit,
                    "top5_correct": top5_hit,
                    "candidate_rows": top_rows,
                    "prompt": str(example.get("prompt") or build_prompt(example, max_context_moves=max_context_moves)),
                }
            )

    total = len(eval_examples)
    return {
        "top1": (top1_correct / total) if total else 0.0,
        "top5": (top5_correct / total) if total else 0.0,
        "total": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "by_side": _finalize_bucket_stats(by_side),
        "by_phase": _finalize_bucket_stats(by_phase),
        "debug_rows": debug_rows,
    }


def pick_top_wrong_moves(
    ranked_rows: Sequence[Mapping[str, object]],
    target_move: str,
    *,
    limit: int,
) -> list[str]:
    wrong: list[str] = []
    for row in ranked_rows:
        move = str(row["norm_move"])
        if move == target_move:
            continue
        wrong.append(move)
        if len(wrong) >= limit:
            break
    return wrong


def select_reranker_negatives(
    example: Mapping[str, object],
    *,
    rng,
    hard_negative_moves: Iterable[str] | None = None,
    uniform_count: int = 4,
    hard_count: int = 3,
) -> list[str]:
    target = str(example["norm_target"])
    legal_moves = [str(move) for move in example["legal_moves_norm"] if str(move) != target]

    selected: list[str] = []
    if hard_negative_moves is not None:
        for move in hard_negative_moves:
            move = str(move)
            if move == target or move not in legal_moves or move in selected:
                continue
            selected.append(move)
            if len(selected) >= hard_count:
                break

    remaining = [move for move in legal_moves if move not in selected]
    if remaining:
        sample_n = min(uniform_count, len(remaining))
        selected.extend(rng.sample(remaining, sample_n))
        remaining = [move for move in remaining if move not in selected]

    while len(selected) < (uniform_count + hard_count) and remaining:
        choice = rng.choice(remaining)
        selected.append(choice)
        remaining = [move for move in remaining if move != choice]

    return selected
