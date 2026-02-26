import os
import re
import random
from typing import Optional

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "daavidhauser/chess-bot-3000-250m"
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


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
        move = _extract_uci(text[len(prompt):] if prompt else text)
        if move in legal:
            return move

    return random.choice(list(legal))
