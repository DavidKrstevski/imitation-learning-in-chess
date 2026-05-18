import os
import random
import re
from typing import Optional

import chess
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.player_policy import (
    DEFAULT_MAX_CONTEXT_MOVES,
    build_prompt,
    normalize_board,
    normalize_move_uci,
    score_legal_moves,
)
from backend.training_service import get_player_model_dir

MODEL_ID = "daavidhauser/chess-bot-3000-250m"
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
_PLAYER_MODEL_CACHE: dict[str, tuple[object, object, str]] = {}


def load_model(model_id: str = MODEL_ID):
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _extract_uci(text: str) -> Optional[str]:
    m = _UCI_RE.search(text.lower())
    return m.group(1) if m else None


def choose_model_move(board, tokenizer, model, device: str, tries: int = 12) -> str:
    legal = {m.uci() for m in board.legal_moves}
    prompt = " ".join(m.uci() for m in board.move_stack)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(tries):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        move = _extract_uci(text[len(prompt) :] if prompt else text)
        if move in legal:
            return move

    return random.choice(list(legal))


def _load_player_model(player_id: str):
    cached = _PLAYER_MODEL_CACHE.get(player_id)
    if cached is not None:
        return cached

    model_dir = get_player_model_dir(player_id)
    base_model_path = model_dir / "base_model.txt"
    model_id = base_model_path.read_text(encoding="utf-8").strip() if base_model_path.exists() else MODEL_ID
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
    model = PeftModel.from_pretrained(base_model, model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    _PLAYER_MODEL_CACHE.clear()
    _PLAYER_MODEL_CACHE[player_id] = (tokenizer, model, device)
    return tokenizer, model, device


def _live_player_example(board: chess.Board) -> dict:
    player_side = "white" if board.turn == chess.WHITE else "black"
    norm_board = normalize_board(board, player_side)
    orig_context_moves = [move.uci() for move in board.move_stack]
    norm_context_moves = [normalize_move_uci(move, player_side) for move in orig_context_moves]
    legal_moves_norm = [normalize_move_uci(move.uci(), player_side) for move in board.legal_moves]
    context_tail = norm_context_moves[-DEFAULT_MAX_CONTEXT_MOVES:]
    example = {
        "side": player_side,
        "norm_fen": norm_board.fen(),
        "norm_context_moves": norm_context_moves,
        "norm_context_tail_moves": context_tail,
        "legal_moves_norm": legal_moves_norm,
    }
    example["prompt"] = build_prompt(example, max_context_moves=DEFAULT_MAX_CONTEXT_MOVES)
    return example


def choose_player_model_move(board: chess.Board, player_id: str) -> str:
    tokenizer, model, device = _load_player_model(player_id)
    example = _live_player_example(board)
    ranked = score_legal_moves(
        model=model,
        tokenizer=tokenizer,
        example=example,
        device=device,
        batch_size=64,
        max_context_moves=DEFAULT_MAX_CONTEXT_MOVES,
    )
    legal = {move.uci() for move in board.legal_moves}
    for row in ranked:
        move = str(row["orig_move"])
        if move in legal:
            return move

    for row in ranked:
        move = str(row["orig_move"])
        if len(move) == 4:
            promotion = f"{move}q"
            if promotion in legal:
                return promotion

    return random.choice(list(legal))
