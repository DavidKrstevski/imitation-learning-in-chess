from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import shutil
import traceback
from bisect import bisect_left
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import requests
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class ExperimentConfig:
    player_usernames: tuple[str, ...] = (
        "Vlad_Lazarev79",
        "ChessTheory64",
        "username1900",
        "RubiRedhead",
        "UniversalRuler",
        "SuNYUnJiN",
        "Dr_Labubu",
    )
    perf_type: str = "classical"
    max_games: int = 500
    rated_only: bool = False
    split_seed: int = 42
    split_strategy: str = "chronological"
    test_frac: float = 0.2
    val_frac_within_train: float = 0.2
    train_games_per_player: int = 100
    min_context_ply: int = 10
    model_id: str = "daavidhauser/chess-bot-3000-250m"
    max_length: int = 256
    candidate_scoring_batch_size: int = 64
    top_ks: tuple[int, ...] = (1, 3, 5, 10)
    debug_examples: int = 10
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 4 if torch.cuda.is_available() else 1
    per_device_eval_batch_size: int = 4 if torch.cuda.is_available() else 1
    logging_steps: int = 25
    save_steps: int = 1000
    save_total_limit: int = 2
    results_root_name: str = "results"
    require_train_games_per_player: bool = False
    learning_curve_train_games: tuple[int, ...] = ()
    learning_curve_player_username: str | None = None
    learning_curve_save_models: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def ensure_root_dirs(run_id: str, results_root_name: str) -> dict[str, Path]:
    root = Path(results_root_name) / run_id
    dirs = {
        "root": root,
        "players": root / "players",
        "plots": root / "plots",
    }
    for path_obj in dirs.values():
        path_obj.mkdir(parents=True, exist_ok=True)
    return dirs


def root_dirs_from_existing_results(results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir)
    dirs = {
        "root": root,
        "players": root / "players",
        "plots": root / "plots",
    }
    for path_obj in dirs.values():
        path_obj.mkdir(parents=True, exist_ok=True)
    return dirs


def experiment_config_from_payload(payload: dict[str, Any]) -> ExperimentConfig:
    field_names = {field.name for field in fields(ExperimentConfig)}
    config_kwargs = {key: value for key, value in payload.items() if key in field_names}
    if "player_usernames" in config_kwargs:
        config_kwargs["player_usernames"] = tuple(config_kwargs["player_usernames"])
    if "top_ks" in config_kwargs:
        config_kwargs["top_ks"] = tuple(int(top_k) for top_k in config_kwargs["top_ks"])
    if "learning_curve_train_games" in config_kwargs:
        config_kwargs["learning_curve_train_games"] = tuple(
            int(game_count) for game_count in config_kwargs["learning_curve_train_games"]
        )
    return ExperimentConfig(**config_kwargs)


def ensure_player_dirs(root_dirs: dict[str, Path], username: str) -> dict[str, Path]:
    player_root = root_dirs["players"] / slugify_name(username)
    dirs = {
        "root": player_root,
        "scratch": player_root / "scratch",
        "best_model": player_root / "best_model",
    }
    for path_obj in dirs.values():
        path_obj.mkdir(parents=True, exist_ok=True)
    return dirs


def write_player_status(
    player_dirs: dict[str, Path],
    username: str,
    stage: str,
    **details: Any,
) -> None:
    payload = {
        "username": username,
        "stage": stage,
        "updated_at": utc_timestamp(),
        **details,
    }
    write_json(player_dirs["root"] / "status.json", payload)


def cleanup_torch_objects(*objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue
        try:
            obj.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def maybe_login_hf() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Hugging Face token detected.")
    else:
        print("No HF token found. Proceeding without explicit login.")
    return token


def min_total_games_required(config: ExperimentConfig) -> int:
    # train_games_per_player is a cap. Players with fewer train games should still run.
    return 10


def load_lichess_user_profile(username: str, perf_type: str) -> dict[str, Any]:
    response = requests.get(
        f"https://lichess.org/api/user/{username}",
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    perf = (payload.get("perfs") or {}).get(perf_type, {})
    return {
        "username": payload.get("username") or username,
        "title": payload.get("title"),
        "country": payload.get("profile", {}).get("country"),
        "elo": perf.get("rating"),
        "games": perf.get("games"),
        "raw_profile": payload,
    }


def load_lichess_games_san(
    username: str,
    max_games: int,
    perf_type: str,
    rated_only: bool,
) -> list[dict[str, Any]]:
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {"Accept": "application/x-ndjson"}
    page_size = min(max_games, 10000)
    raw_games = []
    seen_ids = set()
    until = None

    while len(raw_games) < max_games:
        current_max = min(page_size, max_games - len(raw_games))
        params = {
            "max": current_max,
            "moves": "true",
            "pgnInJson": "false",
            "opening": "false",
            "clocks": "false",
            "evals": "false",
            "perfType": perf_type,
            "sort": "dateDesc",
        }
        if until is not None:
            params["until"] = until
        if rated_only:
            params["rated"] = "true"

        page_rows = []
        response = requests.get(url, headers=headers, params=params, stream=True, timeout=(30, 300))
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            game = json.loads(line)
            game_id = game.get("id")
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)
            page_rows.append(game)

        if not page_rows:
            break

        raw_games.extend(page_rows)
        oldest_created_at = min((game.get("createdAt") or 0) for game in page_rows)
        if oldest_created_at <= 0 or len(page_rows) < current_max:
            break
        until = oldest_created_at - 1

    games = []
    for idx, game in enumerate(raw_games):
        game_perf = game.get("perf") or game.get("speed")
        if game_perf != perf_type:
            continue
        games.append(
            {
                "id": game.get("id") or f"game_{idx}",
                "perf": game_perf,
                "rated": game.get("rated"),
                "created_at": game.get("createdAt") or idx,
                "white": game.get("players", {}).get("white", {}).get("user", {}).get("name"),
                "black": game.get("players", {}).get("black", {}).get("user", {}).get("name"),
                "white_rating": game.get("players", {}).get("white", {}).get("rating"),
                "black_rating": game.get("players", {}).get("black", {}).get("rating"),
                "moves_san": game.get("moves", "").split(),
            }
        )
    return games


def parse_target_games(raw_games: list[dict[str, Any]], username: str) -> list[dict[str, Any]]:
    user_lower = username.lower()
    parsed_games = []
    for game in raw_games:
        white = (game.get("white") or "").lower()
        black = (game.get("black") or "").lower()
        san_moves = game.get("moves_san", [])

        if white == user_lower:
            user_color = chess.WHITE
        elif black == user_lower:
            user_color = chess.BLACK
        else:
            continue

        board = chess.Board()
        uci_moves = []
        valid_game = True
        for san in san_moves:
            try:
                move = board.parse_san(san)
            except Exception:
                valid_game = False
                break
            uci_moves.append(move.uci())
            board.push(move)

        if valid_game and len(uci_moves) >= 2:
            parsed_games.append(
                {
                    "id": game["id"],
                    "created_at": game.get("created_at"),
                    "user_color": "white" if user_color == chess.WHITE else "black",
                    "white_rating": game.get("white_rating"),
                    "black_rating": game.get("black_rating"),
                    "uci_moves": uci_moves,
                }
            )
    return parsed_games


def split_games_train_val_test(
    game_rows: list[dict[str, Any]],
    split_seed: int,
    split_strategy: str,
    test_frac: float,
    val_frac_within_train: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(game_rows)
    if len(rows) < 10:
        raise ValueError("Need at least 10 games for a stable train/val/test split.")

    if split_strategy == "chronological":
        rows.sort(key=lambda row: (row.get("created_at") or 0, row["id"]))
    elif split_strategy == "random":
        rng = random.Random(split_seed)
        rng.shuffle(rows)
    else:
        raise ValueError("split_strategy must be 'chronological' or 'random'")

    test_n = max(1, int(round(len(rows) * test_frac)))
    train_val_rows = rows[:-test_n]
    test_rows = rows[-test_n:]

    val_n = max(1, int(round(len(train_val_rows) * val_frac_within_train)))
    train_rows = train_val_rows[:-val_n]
    val_rows = train_val_rows[-val_n:]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Split produced an empty train, val, or test partition.")

    return train_rows, val_rows, test_rows


def select_fixed_train_games(
    train_rows: list[dict[str, Any]],
    train_games_per_player: int,
    split_strategy: str,
    split_seed: int,
    username: str,
    require_train_games_per_player: bool = False,
) -> list[dict[str, Any]]:
    if len(train_rows) < train_games_per_player:
        if require_train_games_per_player:
            raise ValueError(
                f"{username} has only {len(train_rows)} train games after splitting, "
                f"but require_train_games_per_player=True needs {train_games_per_player}."
            )
        print(
            f"{username} has only {len(train_rows)} train games after splitting; "
            f"using all available train games instead of {train_games_per_player}."
        )
        return list(train_rows)

    if split_strategy == "chronological":
        return train_rows[-train_games_per_player:]

    rng = random.Random(split_seed)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:train_games_per_player])
    return [train_rows[idx] for idx in selected]


def bucket_elo(rating: int | None) -> int:
    if rating is None:
        return 1500
    bucketed = int(round(float(rating) / 100.0) * 100)
    return max(0, min(3500, bucketed))


def build_david_context(game: dict[str, Any], ply_idx: int) -> str:
    white_token = f"<WHITE:{bucket_elo(game.get('white_rating'))}>"
    black_token = f"<BLACK:{bucket_elo(game.get('black_rating'))}>"
    prefix_parts = ["<BOG>", white_token, black_token]
    move_parts = game["uci_moves"][:ply_idx]
    return " ".join(prefix_parts + move_parts)


def player_move_number(side: str, ply_idx: int) -> int:
    if (side or "").lower() == "white":
        return (int(ply_idx) // 2) + 1
    if (side or "").lower() == "black":
        return ((int(ply_idx) + 1) // 2)
    return (int(ply_idx) // 2) + 1


def build_examples_from_games(game_rows: list[dict[str, Any]], min_context_ply: int) -> list[dict[str, Any]]:
    examples = []
    for game in game_rows:
        user_is_white = game["user_color"] == "white"
        moves = game["uci_moves"]
        board = chess.Board()

        for ply_idx, move_uci in enumerate(moves):
            user_to_move = (board.turn == chess.WHITE) if user_is_white else (board.turn == chess.BLACK)
            if user_to_move and ply_idx >= max(0, min_context_ply):
                examples.append(
                    {
                        "game_id": game["id"],
                        "side": game["user_color"],
                        "ply_idx": ply_idx,
                        "player_move_number": player_move_number(game["user_color"], ply_idx),
                        "context": build_david_context(game, ply_idx),
                        "target": move_uci,
                        "legal_moves": [move.uci() for move in board.legal_moves],
                        "white_rating": game.get("white_rating"),
                        "black_rating": game.get("black_rating"),
                    }
                )
            board.push(chess.Move.from_uci(move_uci))
    return examples


def load_tokenizer(model_id: str, token: str | None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token or None, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_id: str,
    tokenizer: AutoTokenizer,
    device: str,
    token: str | None,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token or None, trust_remote_code=True)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def load_saved_player_model(
    player_dir: Path,
    tokenizer: AutoTokenizer,
    device: str,
) -> AutoModelForCausalLM:
    best_model_dir = player_dir / "best_model"
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Missing best_model directory in {player_dir}")
    model = AutoModelForCausalLM.from_pretrained(str(best_model_dir), trust_remote_code=True)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def get_prompt_ids(tokenizer: AutoTokenizer, context: str) -> list[int]:
    prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if prompt_ids:
        return prompt_ids
    fallback_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if fallback_id is not None:
        return [fallback_id]
    fallback_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    if fallback_ids:
        return fallback_ids
    raise ValueError("Tokenizer cannot produce a fallback prefix token for empty context")


def build_scoring_batch(
    tokenizer: AutoTokenizer,
    context: str,
    candidate_moves: list[str],
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    prompt_ids = get_prompt_ids(tokenizer, context)
    rows = []
    prompt_lengths = []
    for move in candidate_moves:
        completion_ids = tokenizer(" " + move, add_special_tokens=False)["input_ids"]
        max_prompt_len = max(0, max_length - len(completion_ids))
        trimmed_prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        input_ids = trimmed_prompt_ids + completion_ids
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        prompt_lengths.append(len(trimmed_prompt_ids))
    encoded = tokenizer.pad(rows, padding=True, return_tensors="pt")
    return encoded, prompt_lengths


@torch.no_grad()
def score_candidate_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    candidate_moves: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    encoded, prompt_lengths = build_scoring_batch(tokenizer, context, candidate_moves, max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    scores: dict[str, float] = {}

    for start in range(0, len(candidate_moves), batch_size):
        end = min(start + batch_size, len(candidate_moves))
        batch_input_ids = input_ids[start:end].to(device)
        batch_attention_mask = attention_mask[start:end].to(device)
        labels = batch_input_ids.clone()
        for row_idx, prompt_len in enumerate(prompt_lengths[start:end]):
            labels[row_idx, :prompt_len] = -100

        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        loss_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
        token_log_probs = torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        sequence_log_probs = (token_log_probs * loss_mask).sum(dim=-1).detach().cpu().tolist()

        for move, sequence_log_prob in zip(candidate_moves[start:end], sequence_log_probs):
            scores[move] = float(sequence_log_prob)

    return scores


def score_legal_move_distribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: dict[str, Any],
    device: str,
    candidate_scoring_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    candidate_moves = example["legal_moves"]
    raw_scores = score_candidate_log_probs(
        model=model,
        tokenizer=tokenizer,
        context=example["context"],
        candidate_moves=candidate_moves,
        device=device,
        batch_size=candidate_scoring_batch_size,
        max_length=max_length,
    )
    ordered_moves = list(candidate_moves)
    score_tensor = torch.tensor([raw_scores[move] for move in ordered_moves], dtype=torch.float32)
    probability_tensor = torch.softmax(score_tensor, dim=0)
    distribution = {move: float(probability_tensor[idx].item()) for idx, move in enumerate(ordered_moves)}
    ranked_rows = sorted(
        ({"move": move, "score": float(raw_scores[move]), "probability": distribution[move]} for move in ordered_moves),
        key=lambda row: row["probability"],
        reverse=True,
    )
    target_rank = next(idx for idx, row in enumerate(ranked_rows, start=1) if row["move"] == example["target"])
    target_probability = distribution[example["target"]]
    return {
        "distribution": distribution,
        "ranked_rows": ranked_rows,
        "target_rank": target_rank,
        "target_probability": target_probability,
    }


def evaluate_policy_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, Any]],
    device: str,
    top_ks: tuple[int, ...],
    debug_n: int,
    candidate_scoring_batch_size: int,
    max_length: int,
    return_distributions: bool = False,
    return_eval_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    total = 0
    rank_sum = 0.0
    top_hits = {k: 0 for k in top_ks}
    debug_rows = []
    distributions: list[dict[str, Any]] | None = [] if return_distributions else None
    eval_rows: list[dict[str, Any]] | None = [] if return_eval_rows else None

    for example in examples:
        scored = score_legal_move_distribution(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            candidate_scoring_batch_size=candidate_scoring_batch_size,
            max_length=max_length,
        )
        target_rank = scored["target_rank"]
        total += 1
        rank_sum += target_rank
        for top_k in top_ks:
            if target_rank <= top_k:
                top_hits[top_k] += 1
        if len(debug_rows) < debug_n:
            debug_rows.append(
                {
                    "game_id": example["game_id"],
                    "side": example["side"],
                    "ply_idx": example["ply_idx"],
                    "target": example["target"],
                    "rank": target_rank,
                    "target_probability": scored["target_probability"],
                    "context_tail": " ".join(example["context"].split()[-12:]),
                    "top10_candidates": scored["ranked_rows"][:10],
                }
            )
        if return_distributions and distributions is not None:
            distributions.append(
                {
                    "game_id": example["game_id"],
                    "side": example["side"],
                    "ply_idx": example["ply_idx"],
                    "distribution": scored["distribution"],
                }
            )
        if return_eval_rows and eval_rows is not None:
            eval_rows.append(
                {
                    "game_id": example["game_id"],
                    "side": example["side"],
                    "ply_idx": example["ply_idx"],
                    "player_move_number": example.get(
                        "player_move_number",
                        player_move_number(example.get("side", ""), example["ply_idx"]),
                    ),
                    "target": example["target"],
                    "target_rank": target_rank,
                    "target_probability": scored["target_probability"],
                }
            )

    metrics = {f"top{k}_accuracy": top_hits[k] / total for k in top_ks}
    metrics.update({f"top{k}_correct": top_hits[k] for k in top_ks})
    metrics.update({"mean_rank": rank_sum / total if total else 0.0, "total": total, "debug_rows": debug_rows})
    return metrics, distributions, eval_rows


def average_kl_divergence(
    finetuned_distributions: list[dict[str, Any]],
    baseline_distributions: list[dict[str, Any]],
    epsilon: float = 1e-12,
) -> float:
    if len(finetuned_distributions) != len(baseline_distributions):
        raise ValueError("Distribution lists must be aligned to compute KL divergence")
    divergences = []
    for finetuned_row, baseline_row in zip(finetuned_distributions, baseline_distributions):
        finetuned_dist = finetuned_row["distribution"]
        baseline_dist = baseline_row["distribution"]
        divergence = 0.0
        for move, finetuned_prob in finetuned_dist.items():
            baseline_prob = max(baseline_dist.get(move, epsilon), epsilon)
            finetuned_prob = max(finetuned_prob, epsilon)
            divergence += finetuned_prob * math.log(finetuned_prob / baseline_prob)
        divergences.append(divergence)
    return float(sum(divergences) / len(divergences)) if divergences else 0.0


def summarize_top_metrics(metrics: dict[str, Any], top_ks: tuple[int, ...]) -> dict[str, float]:
    summary = {f"top{k}": round(metrics[f"top{k}_accuracy"], 4) for k in top_ks}
    summary["mean_rank"] = round(metrics["mean_rank"], 4)
    return summary


def tokenize_for_lm(example: dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> dict[str, list[int]]:
    prompt_ids = get_prompt_ids(tokenizer, example["context"])
    target_ids = tokenizer(" " + example["target"], add_special_tokens=False)["input_ids"]
    max_prompt_len = max(0, max_length - len(target_ids))
    trimmed_prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
    input_ids = trimmed_prompt_ids + target_ids
    labels = ([-100] * len(trimmed_prompt_ids)) + target_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def prepare_tokenized_dataset(
    examples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> tuple[Dataset, dict[str, Any]]:
    tokenized_rows = [tokenize_for_lm(example, tokenizer, max_length) for example in examples]
    dataset = Dataset.from_list(tokenized_rows)
    lengths = [len(row["input_ids"]) for row in tokenized_rows]
    tokenization_summary = {
        "num_rows": len(tokenized_rows),
        "max_length": max_length,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "avg_tokens": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    return dataset, tokenization_summary


def make_causal_lm_data_collator(tokenizer: AutoTokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_seq_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_seq_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + ([pad_token_id] * pad_len))
            batch["attention_mask"].append(feature["attention_mask"] + ([0] * pad_len))
            batch["labels"].append(feature["labels"] + ([-100] * pad_len))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collator


def build_lora_model(
    model_id: str,
    tokenizer: AutoTokenizer,
    device: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    token: str | None,
) -> AutoModelForCausalLM:
    base_model = AutoModelForCausalLM.from_pretrained(model_id, token=token or None, trust_remote_code=True)
    if len(tokenizer) != base_model.get_input_embeddings().num_embeddings:
        base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()
    return model


def build_training_args(config: ExperimentConfig, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config.split_seed,
    )


class TrainingProgressCallback(TrainerCallback):
    def __init__(
        self,
        player_dirs: dict[str, Path],
        username: str,
        run_id: str,
        run_name: str,
    ) -> None:
        self.player_dirs = player_dirs
        self.username = username
        self.run_id = run_id
        self.run_name = run_name

    def _write_progress(self, args: TrainingArguments, state: Any, stage: str, logs: dict[str, Any] | None = None) -> None:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        global_step = int(getattr(state, "global_step", 0) or 0)
        progress = {
            "username": self.username,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "stage": stage,
            "updated_at": utc_timestamp(),
            "global_step": global_step,
            "max_steps": max_steps,
            "epoch": getattr(state, "epoch", None),
            "progress": (global_step / max_steps) if max_steps else None,
            "latest_log": logs or {},
            "output_dir": str(args.output_dir),
        }
        write_json(self.player_dirs["root"] / "training_progress.json", progress)
        write_player_status(
            self.player_dirs,
            self.username,
            "training_lora",
            run_id=self.run_id,
            global_step=global_step,
            max_steps=max_steps,
            epoch=getattr(state, "epoch", None),
            progress=progress["progress"],
        )

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._write_progress(args, state, "train_begin")

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._write_progress(args, state, "log", logs)

    def on_save(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._write_progress(args, state, "checkpoint_saved")

    def on_train_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._write_progress(args, state, "train_end")


def train_lora_model(
    config: ExperimentConfig,
    tokenizer: AutoTokenizer,
    train_examples: list[dict[str, Any]],
    model_id: str,
    player_dirs: dict[str, Path],
    run_name: str,
    device: str,
    hf_token: str | None,
    username: str,
    run_id: str,
) -> tuple[AutoModelForCausalLM, dict[str, Any]]:
    scratch_dir = player_dirs["scratch"] / run_name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = get_last_checkpoint(str(scratch_dir))

    train_dataset, tokenization_summary = prepare_tokenized_dataset(train_examples, tokenizer, config.max_length)
    model = build_lora_model(
        model_id=model_id,
        tokenizer=tokenizer,
        device=device,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        token=hf_token,
    )
    training_args = build_training_args(config, str(scratch_dir))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_causal_lm_data_collator(tokenizer),
        callbacks=[TrainingProgressCallback(player_dirs, username, run_id, run_name)],
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    model.eval()

    train_summary = {
        "run_name": run_name,
        "resumed_from_checkpoint": resume_checkpoint,
        "train_examples": len(train_examples),
        "tokenization_summary": tokenization_summary,
        "train_result_metrics": to_builtin(train_result.metrics),
        "trainer_log_history": to_builtin(trainer.state.log_history),
        "training_args": {
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "weight_decay": config.weight_decay,
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
        },
    }

    del trainer
    remove_path(scratch_dir)
    cleanup_torch_objects()
    return model, train_summary


def save_player_model(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    player_dirs: dict[str, Path],
    player_summary: dict[str, Any],
) -> None:
    best_model_dir = player_dirs["best_model"]
    remove_path(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    (best_model_dir / "base_model.txt").write_text(config.model_id, encoding="utf-8")
    write_json(best_model_dir / "player_summary.json", player_summary)


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(median),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_aggregate_metrics(successful_rows: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    if not successful_rows:
        return {}

    test_examples_total = sum(int(row["test_examples"]) for row in successful_rows)
    aggregate = {
        "player_count": len(successful_rows),
        "test_examples_total": test_examples_total,
        "elo_summary": summarize_numeric([float(row["elo"]) for row in successful_rows if row["elo"] is not None]),
    }

    for top_k in top_ks:
        baseline_key = f"baseline_top{top_k}_accuracy"
        finetuned_key = f"finetuned_top{top_k}_accuracy"
        delta_key = f"delta_top{top_k}_accuracy"
        baseline_correct_key = f"baseline_top{top_k}_correct"
        finetuned_correct_key = f"finetuned_top{top_k}_correct"

        aggregate[f"mean_{baseline_key}"] = float(sum(row[baseline_key] for row in successful_rows) / len(successful_rows))
        aggregate[f"mean_{finetuned_key}"] = float(sum(row[finetuned_key] for row in successful_rows) / len(successful_rows))
        aggregate[f"mean_{delta_key}"] = float(sum(row[delta_key] for row in successful_rows) / len(successful_rows))
        aggregate[f"weighted_{baseline_key}"] = (
            float(sum(row[baseline_correct_key] for row in successful_rows) / test_examples_total) if test_examples_total else 0.0
        )
        aggregate[f"weighted_{finetuned_key}"] = (
            float(sum(row[finetuned_correct_key] for row in successful_rows) / test_examples_total) if test_examples_total else 0.0
        )
        aggregate[f"weighted_{delta_key}"] = (
            aggregate[f"weighted_{finetuned_key}"] - aggregate[f"weighted_{baseline_key}"]
        )

    aggregate["mean_baseline_mean_rank"] = float(
        sum(row["baseline_mean_rank"] for row in successful_rows) / len(successful_rows)
    )
    aggregate["mean_finetuned_mean_rank"] = float(
        sum(row["finetuned_mean_rank"] for row in successful_rows) / len(successful_rows)
    )
    aggregate["mean_delta_mean_rank"] = float(sum(row["delta_mean_rank"] for row in successful_rows) / len(successful_rows))
    aggregate["mean_average_kl_finetuned_vs_baseline"] = float(
        sum(row["average_kl_finetuned_vs_baseline"] for row in successful_rows) / len(successful_rows)
    )
    return aggregate


def plot_elo_vs_top1_accuracy(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    perf_type: str,
) -> Path | None:
    elo_rows = [row for row in successful_rows if row["elo"] is not None]
    if not elo_rows:
        return None

    elo_rows = sorted(elo_rows, key=lambda row: (row["elo"], row["username"].lower()))
    elos = [row["elo"] for row in elo_rows]
    baseline_top1 = [row["baseline_top1_accuracy"] for row in elo_rows]
    finetuned_top1 = [row["finetuned_top1_accuracy"] for row in elo_rows]

    plt.figure(figsize=(10, 6))
    for row in elo_rows:
        plt.plot(
            [row["elo"], row["elo"]],
            [row["baseline_top1_accuracy"], row["finetuned_top1_accuracy"]],
            color="tab:gray",
            alpha=0.5,
            linewidth=1.2,
        )
    plt.scatter(elos, baseline_top1, label="Baseline top-1", s=60, color="tab:blue")
    plt.scatter(elos, finetuned_top1, label="Fine-tuned top-1", s=60, color="tab:orange")
    for row in elo_rows:
        plt.annotate(row["username"], (row["elo"], row["finetuned_top1_accuracy"]), fontsize=8, alpha=0.85)
    plt.xlabel(f"{perf_type.title()} ELO")
    plt.ylabel("Top-1 Accuracy")
    plt.title("ELO vs Top-1 Accuracy Before and After Fine-Tuning")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = root_dirs["plots"] / "elo_vs_top1_accuracy.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_elo_vs_delta_top1_accuracy(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    perf_type: str,
) -> Path | None:
    elo_rows = [row for row in successful_rows if row["elo"] is not None]
    if not elo_rows:
        return None

    elo_rows = sorted(elo_rows, key=lambda row: (row["elo"], row["username"].lower()))
    rolling_window = min(9, len(elo_rows))
    if rolling_window % 2 == 0:
        rolling_window -= 1
    half_window = rolling_window // 2
    rolling_rows: list[dict[str, float]] = []
    if rolling_window >= 3:
        for idx in range(len(elo_rows)):
            start = max(0, idx - half_window)
            end = min(len(elo_rows), idx + half_window + 1)
            window_rows = elo_rows[start:end]
            rolling_rows.append(
                {
                    "elo": sum(float(row["elo"]) for row in window_rows) / len(window_rows),
                    "delta": sum(float(row["delta_top1_accuracy"]) for row in window_rows) / len(window_rows),
                }
            )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        [row["elo"] for row in elo_rows],
        [row["delta_top1_accuracy"] for row in elo_rows],
        color="tab:green",
        s=60,
        alpha=0.78,
        label="Players",
    )
    if rolling_rows:
        plt.plot(
            [row["elo"] for row in rolling_rows],
            [row["delta"] for row in rolling_rows],
            color="tab:orange",
            linewidth=2.6,
            marker="o",
            markersize=4,
            label=f"Rolling average ({rolling_window} players)",
        )
    for row in elo_rows:
        plt.annotate(row["username"], (row["elo"], row["delta_top1_accuracy"]), fontsize=8, alpha=0.85)
    plt.xlabel(f"{perf_type.title()} ELO")
    plt.ylabel("Delta Top-1 Accuracy")
    plt.title("ELO vs Top-1 Accuracy Gain")
    plt.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = root_dirs["plots"] / "elo_vs_delta_top1_accuracy.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_players_sorted_by_elo_top1_accuracy(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
) -> Path | None:
    if not successful_rows:
        return None

    rows = sorted(
        successful_rows,
        key=lambda row: (row["elo"] is None, row["elo"] if row["elo"] is not None else 10**9, row["username"].lower()),
    )
    player_labels = [row["username"] for row in rows]
    x_positions = list(range(len(rows)))

    plt.figure(figsize=(max(10, len(rows) * 1.3), 6))
    plt.plot(x_positions, [row["baseline_top1_accuracy"] for row in rows], marker="o", linewidth=2, label="Baseline top-1")
    plt.plot(
        x_positions,
        [row["finetuned_top1_accuracy"] for row in rows],
        marker="o",
        linewidth=2,
        label="Fine-tuned top-1",
    )
    plt.xticks(x_positions, player_labels, rotation=45, ha="right")
    plt.xlabel("Players (sorted by ELO)")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Accuracy by Player")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = root_dirs["plots"] / "players_sorted_by_elo_top1_accuracy.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    half_window = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        stop = min(len(values), idx + half_window + 1)
        smoothed.append(float(sum(values[start:stop]) / (stop - start)))
    return smoothed


def extract_training_loss_points(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    points = []
    for entry in log_history:
        if "loss" not in entry:
            continue
        epoch = entry.get("epoch")
        step = entry.get("step")
        if epoch is None or step is None:
            continue
        points.append(
            {
                "epoch": float(epoch),
                "step": float(step),
                "loss": float(entry["loss"]),
            }
        )
    return points


def interpolate_series(xs: list[float], ys: list[float], target_x: float) -> float:
    if target_x <= xs[0]:
        return ys[0]
    if target_x >= xs[-1]:
        return ys[-1]

    right_idx = bisect_left(xs, target_x)
    left_idx = max(0, right_idx - 1)
    x_left = xs[left_idx]
    x_right = xs[right_idx]
    y_left = ys[left_idx]
    y_right = ys[right_idx]
    if x_right == x_left:
        return y_left
    weight = (target_x - x_left) / (x_right - x_left)
    return y_left + weight * (y_right - y_left)


def load_player_training_curves(successful_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curves = []
    for row in successful_rows:
        player_dir = Path(row["player_results_dir"])
        trainer_logs_path = player_dir / "trainer_logs.json"
        if not trainer_logs_path.exists():
            continue
        trainer_logs = read_json(trainer_logs_path)
        log_history = trainer_logs.get("trainer_log_history") or []
        points = extract_training_loss_points(log_history)
        if not points:
            continue
        train_games_used = row.get("train_games_used")
        if train_games_used is None:
            dataset_summary_path = player_dir / "dataset_summary.json"
            if dataset_summary_path.exists():
                train_games_used = (read_json(dataset_summary_path) or {}).get("train_games_used")
        try:
            train_games_used = int(train_games_used)
        except (TypeError, ValueError):
            train_games_used = 0
        if train_games_used <= 0:
            continue
        curves.append(
            {
                "username": row["username"],
                "elo": row.get("elo"),
                "train_games_used": train_games_used,
                "epochs": [point["epoch"] for point in points],
                "games_seen": [point["epoch"] * train_games_used for point in points],
                "steps": [point["step"] for point in points],
                "losses": [point["loss"] for point in points],
                "final_train_loss": float((trainer_logs.get("train_result_metrics") or {}).get("train_loss", 0.0)),
            }
        )
    return curves


def plot_elo_vs_kl_divergence(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    perf_type: str,
) -> Path | None:
    elo_rows = [row for row in successful_rows if row["elo"] is not None]
    if not elo_rows:
        return None

    elo_rows = sorted(elo_rows, key=lambda row: (row["elo"], row["username"].lower()))
    plt.figure(figsize=(10, 6))
    plt.scatter(
        [row["elo"] for row in elo_rows],
        [row["average_kl_finetuned_vs_baseline"] for row in elo_rows],
        color="tab:purple",
        s=70,
        alpha=0.9,
    )
    for row in elo_rows:
        plt.annotate(
            row["username"],
            (row["elo"], row["average_kl_finetuned_vs_baseline"]),
            fontsize=8,
            alpha=0.85,
        )
    plt.xlabel(f"{perf_type.title()} ELO")
    plt.ylabel("Average KL Divergence")
    plt.title("ELO vs KL Divergence After Fine-Tuning")
    plt.grid(alpha=0.3)
    plot_path = root_dirs["plots"] / "elo_vs_kl_divergence.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_training_loss_by_player(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    smoothing_window: int = 5,
) -> Path | None:
    curves = load_player_training_curves(successful_rows)
    if not curves:
        return None

    plt.figure(figsize=(11, 7))
    for curve in sorted(curves, key=lambda item: (item["elo"] is None, item["elo"] or 0, item["username"].lower())):
        smooth_losses = moving_average(curve["losses"], smoothing_window)
        elo_label = int(curve["elo"]) if curve["elo"] is not None else "n/a"
        label = f"{curve['username']} ({elo_label}, {curve['train_games_used']} games)"
        plt.plot(curve["games_seen"], smooth_losses, linewidth=1.8, alpha=0.9, label=label)
    plt.xlabel("Training games seen (epoch x train games used)")
    plt.ylabel("Training Loss")
    plt.title("Training Loss by Player over Seen Games")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plot_path = root_dirs["plots"] / "training_loss_by_player.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_mean_training_loss(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    smoothing_window: int = 5,
    grid_points: int = 200,
) -> Path | None:
    curves = load_player_training_curves(successful_rows)
    if not curves:
        return None

    min_seen_games = min(curve["games_seen"][0] for curve in curves if curve["games_seen"])
    max_seen_games = max(curve["games_seen"][-1] for curve in curves if curve["games_seen"])
    if max_seen_games <= min_seen_games:
        return None

    grid_points = max(25, int(grid_points))
    games_grid = [
        min_seen_games + ((max_seen_games - min_seen_games) * idx / (grid_points - 1))
        for idx in range(grid_points)
    ]
    available_games_grid = []
    mean_losses = []
    lower_losses = []
    upper_losses = []

    for games_value in games_grid:
        values = [
            interpolate_series(curve["games_seen"], curve["losses"], games_value)
            for curve in curves
            if curve["games_seen"][0] <= games_value <= curve["games_seen"][-1]
        ]
        if not values:
            continue
        available_games_grid.append(games_value)
        mean_losses.append(float(sum(values) / len(values)))
        lower_losses.append(float(min(values)))
        upper_losses.append(float(max(values)))
    if not mean_losses:
        return None

    smoothed_mean = moving_average(mean_losses, smoothing_window)

    plt.figure(figsize=(11, 7))
    for curve in curves:
        light_losses = moving_average(curve["losses"], smoothing_window)
        plt.plot(curve["games_seen"], light_losses, color="tab:gray", alpha=0.18, linewidth=1.0)
    plt.fill_between(available_games_grid, lower_losses, upper_losses, color="tab:blue", alpha=0.12, label="Min-max range")
    plt.plot(available_games_grid, smoothed_mean, color="tab:blue", linewidth=2.5, label="Mean loss")
    plt.xlabel("Training games seen (epoch x train games used)")
    plt.ylabel("Training Loss")
    plt.title("Mean Training Loss Across Players over Seen Games")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = root_dirs["plots"] / "training_loss_mean.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def summarize_mean_training_loss_progress(
    successful_rows: list[dict[str, Any]],
    interval_games: int = 50,
    smoothing_window: int = 5,
    grid_points: int = 400,
) -> dict[str, Any]:
    curves = load_player_training_curves(successful_rows)
    if not curves:
        return {
            "interval_games": interval_games,
            "start_loss": None,
            "best_loss": None,
            "best_loss_games_seen": None,
            "progress_rows": [],
        }

    min_seen_games = min(curve["games_seen"][0] for curve in curves if curve["games_seen"])
    max_seen_games = max(curve["games_seen"][-1] for curve in curves if curve["games_seen"])
    if max_seen_games <= min_seen_games:
        return {
            "interval_games": interval_games,
            "start_loss": None,
            "best_loss": None,
            "best_loss_games_seen": None,
            "progress_rows": [],
        }

    grid_points = max(50, int(grid_points))
    games_grid = [
        min_seen_games + ((max_seen_games - min_seen_games) * idx / (grid_points - 1))
        for idx in range(grid_points)
    ]
    available_games_grid = []
    mean_losses = []
    contributing_players = []

    for games_value in games_grid:
        values = [
            interpolate_series(curve["games_seen"], curve["losses"], games_value)
            for curve in curves
            if curve["games_seen"][0] <= games_value <= curve["games_seen"][-1]
        ]
        if not values:
            continue
        available_games_grid.append(games_value)
        mean_losses.append(float(sum(values) / len(values)))
        contributing_players.append(len(values))

    if not mean_losses:
        return {
            "interval_games": interval_games,
            "start_loss": None,
            "best_loss": None,
            "best_loss_games_seen": None,
            "progress_rows": [],
        }

    smoothed_mean = moving_average(mean_losses, smoothing_window)
    start_loss = float(smoothed_mean[0])
    best_idx = min(range(len(smoothed_mean)), key=lambda idx: smoothed_mean[idx])
    best_loss = float(smoothed_mean[best_idx])
    best_loss_games_seen = float(available_games_grid[best_idx])
    max_loss_reduction = max(0.0, start_loss - best_loss)

    progress_rows = []
    step = max(1, int(interval_games))
    max_sample_games = int(math.floor(max_seen_games / step) * step)
    best_loss_so_far = start_loss
    for games_seen in range(step, max_sample_games + 1, step):
        mean_loss = float(interpolate_series(available_games_grid, smoothed_mean, float(games_seen)))
        best_loss_so_far = min(best_loss_so_far, mean_loss)
        contributing_count = 0
        for idx, grid_games_seen in enumerate(available_games_grid):
            if grid_games_seen >= games_seen:
                contributing_count = contributing_players[idx]
                break
        if not contributing_count:
            contributing_count = contributing_players[-1]

        if max_loss_reduction > 0:
            progress_pct = ((start_loss - best_loss_so_far) / max_loss_reduction) * 100.0
        else:
            progress_pct = 100.0
        progress_rows.append(
            {
                "games_seen": int(games_seen),
                "mean_training_loss": mean_loss,
                "best_mean_training_loss_so_far": float(best_loss_so_far),
                "progress_pct_of_max_loss_reduction": float(max(0.0, min(100.0, progress_pct))),
                "contributing_players": int(contributing_count),
            }
        )

    return {
        "interval_games": step,
        "start_loss": start_loss,
        "best_loss": best_loss,
        "best_loss_games_seen": best_loss_games_seen,
        "progress_rows": progress_rows,
    }


def summarize_mean_training_loss_quality(
    successful_rows: list[dict[str, Any]],
    interval_games: int = 100,
    smoothing_window: int = 5,
    grid_points: int = 400,
) -> dict[str, Any]:
    curves = load_player_training_curves(successful_rows)
    if not curves:
        return {
            "interval_games": interval_games,
            "player_count": 0,
            "common_max_games_seen": None,
            "worst_loss": None,
            "best_loss": None,
            "worst_loss_games_seen": None,
            "best_loss_games_seen": None,
            "quality_rows": [],
        }

    min_seen_games = min(curve["games_seen"][0] for curve in curves if curve["games_seen"])
    max_seen_games = max(curve["games_seen"][-1] for curve in curves if curve["games_seen"])
    common_max_games_seen = min(curve["games_seen"][-1] for curve in curves if curve["games_seen"])
    if max_seen_games <= min_seen_games:
        return {
            "interval_games": interval_games,
            "player_count": len(curves),
            "common_max_games_seen": common_max_games_seen,
            "worst_loss": None,
            "best_loss": None,
            "worst_loss_games_seen": None,
            "best_loss_games_seen": None,
            "quality_rows": [],
        }

    grid_points = max(50, int(grid_points))
    games_grid = [
        min_seen_games + ((max_seen_games - min_seen_games) * idx / (grid_points - 1))
        for idx in range(grid_points)
    ]
    available_games_grid = []
    mean_losses = []
    contributing_players = []

    for games_value in games_grid:
        values = [
            interpolate_series(curve["games_seen"], curve["losses"], games_value)
            for curve in curves
            if curve["games_seen"][0] <= games_value <= curve["games_seen"][-1]
        ]
        if not values:
            continue
        available_games_grid.append(games_value)
        mean_losses.append(float(sum(values) / len(values)))
        contributing_players.append(len(values))

    restricted_rows = [
        (games_value, mean_loss, player_count)
        for games_value, mean_loss, player_count in zip(available_games_grid, mean_losses, contributing_players)
        if games_value <= common_max_games_seen and player_count == len(curves)
    ]
    available_games_grid = [games_value for games_value, _, _ in restricted_rows]
    mean_losses = [mean_loss for _, mean_loss, _ in restricted_rows]
    contributing_players = [player_count for _, _, player_count in restricted_rows]

    if not mean_losses:
        return {
            "interval_games": interval_games,
            "player_count": len(curves),
            "common_max_games_seen": common_max_games_seen,
            "worst_loss": None,
            "best_loss": None,
            "worst_loss_games_seen": None,
            "best_loss_games_seen": None,
            "quality_rows": [],
        }

    smoothed_mean = moving_average(mean_losses, smoothing_window)
    worst_idx = max(range(len(smoothed_mean)), key=lambda idx: smoothed_mean[idx])
    best_idx = min(range(len(smoothed_mean)), key=lambda idx: smoothed_mean[idx])
    worst_loss = float(smoothed_mean[worst_idx])
    best_loss = float(smoothed_mean[best_idx])
    loss_span = max(0.0, worst_loss - best_loss)

    quality_rows = []
    step = max(1, int(interval_games))
    max_sample_games = int(math.floor(common_max_games_seen / step) * step)
    for games_seen in range(step, max_sample_games + 1, step):
        mean_loss = float(interpolate_series(available_games_grid, smoothed_mean, float(games_seen)))
        contributing_count = 0
        for idx, grid_games_seen in enumerate(available_games_grid):
            if grid_games_seen >= games_seen:
                contributing_count = contributing_players[idx]
                break
        if not contributing_count:
            contributing_count = contributing_players[-1]

        if loss_span > 0:
            quality_pct = ((worst_loss - mean_loss) / loss_span) * 100.0
        else:
            quality_pct = 100.0
        quality_rows.append(
            {
                "games_seen": int(games_seen),
                "mean_training_loss": mean_loss,
                "normalized_quality_pct": float(max(0.0, min(100.0, quality_pct))),
                "contributing_players": int(contributing_count),
            }
        )

    return {
        "interval_games": step,
        "player_count": len(curves),
        "common_max_games_seen": common_max_games_seen,
        "worst_loss": worst_loss,
        "best_loss": best_loss,
        "worst_loss_games_seen": float(available_games_grid[worst_idx]),
        "best_loss_games_seen": float(available_games_grid[best_idx]),
        "quality_rows": quality_rows,
    }


def plot_mean_training_loss_quality(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    interval_games: int = 100,
    smoothing_window: int = 5,
    grid_points: int = 400,
) -> Path | None:
    quality_summary = summarize_mean_training_loss_quality(
        successful_rows=successful_rows,
        interval_games=interval_games,
        smoothing_window=smoothing_window,
        grid_points=grid_points,
    )
    quality_rows = quality_summary.get("quality_rows") or []
    if not quality_rows:
        return None

    games_values = [row["games_seen"] for row in quality_rows]
    quality_values = [row["normalized_quality_pct"] for row in quality_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(games_values, quality_values, color="tab:green", linewidth=2.2, marker="o", markersize=6)
    for row in quality_rows:
        plt.annotate(
            f"n={row['contributing_players']}",
            (row["games_seen"], row["normalized_quality_pct"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            alpha=0.85,
        )
    plt.xlabel(f"Training games seen (every {quality_summary['interval_games']} games)")
    plt.ylabel("Normalized training quality (%)")
    plt.title("Normalized Mean Training Quality (Common Player Range)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plot_path = root_dirs["plots"] / "training_loss_quality_100_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def load_player_eval_rows_artifact(
    player_dir: Path,
    artifact_name: str,
) -> list[dict[str, Any]] | None:
    artifact_path = player_dir / artifact_name
    if not artifact_path.exists():
        return None
    payload = read_json(artifact_path)
    if not isinstance(payload, list):
        return None
    return payload


def aggregate_target_probability_by_player_move(
    eval_rows: list[dict[str, Any]],
    zero_based_player_move_number: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float]] = {}
    for row in eval_rows:
        probability = row.get("target_probability")
        ply_idx = row.get("ply_idx")
        if probability is None or ply_idx is None:
            continue
        move_number = int(
            row.get("player_move_number")
            or player_move_number(str(row.get("side") or ""), int(ply_idx))
        )
        if zero_based_player_move_number:
            move_number = max(0, move_number - 1)
        bucket = buckets.setdefault(move_number, {"sum_probability": 0.0, "count": 0.0})
        bucket["sum_probability"] += float(probability)
        bucket["count"] += 1.0
    return [
        {
            "player_move_number": move_number,
            "average_target_probability": bucket["sum_probability"] / bucket["count"],
            "count": int(bucket["count"]),
        }
        for move_number, bucket in sorted(buckets.items())
        if bucket["count"] > 0
    ]


def collect_eval_rows_from_successful_players(
    successful_rows: list[dict[str, Any]],
    artifact_name: str,
) -> list[dict[str, Any]] | None:
    combined_rows: list[dict[str, Any]] = []
    for row in successful_rows:
        player_dir = Path(row["player_results_dir"])
        eval_rows = load_player_eval_rows_artifact(player_dir, artifact_name)
        if eval_rows is None:
            return None
        combined_rows.extend(eval_rows)
    return combined_rows


def aggregate_delta_target_probability_by_player_move(
    successful_rows: list[dict[str, Any]],
    baseline_artifact_name: str = "baseline_eval_rows.json",
    finetuned_artifact_name: str = "finetuned_eval_rows.json",
    zero_based_player_move_number: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float]] = {}
    for row in successful_rows:
        player_dir = Path(row["player_results_dir"])
        baseline_rows = load_player_eval_rows_artifact(player_dir, baseline_artifact_name)
        finetuned_rows = load_player_eval_rows_artifact(player_dir, finetuned_artifact_name)
        if baseline_rows is None or finetuned_rows is None:
            return []

        baseline_by_key = {
            (str(item.get("game_id") or ""), str(item.get("side") or ""), int(item.get("ply_idx") or -1)): item
            for item in baseline_rows
        }
        finetuned_by_key = {
            (str(item.get("game_id") or ""), str(item.get("side") or ""), int(item.get("ply_idx") or -1)): item
            for item in finetuned_rows
        }

        for key in sorted(set(baseline_by_key) & set(finetuned_by_key)):
            baseline_item = baseline_by_key[key]
            finetuned_item = finetuned_by_key[key]
            baseline_probability = baseline_item.get("target_probability")
            finetuned_probability = finetuned_item.get("target_probability")
            ply_idx = finetuned_item.get("ply_idx")
            if baseline_probability is None or finetuned_probability is None or ply_idx is None:
                continue
            move_number = int(
                finetuned_item.get("player_move_number")
                or player_move_number(str(finetuned_item.get("side") or ""), int(ply_idx))
            )
            if zero_based_player_move_number:
                move_number = max(0, move_number - 1)
            bucket = buckets.setdefault(move_number, {"sum_delta": 0.0, "count": 0.0})
            bucket["sum_delta"] += float(finetuned_probability) - float(baseline_probability)
            bucket["count"] += 1.0

    return [
        {
            "player_move_number": move_number,
            "average_delta_target_probability": bucket["sum_delta"] / bucket["count"],
            "count": int(bucket["count"]),
        }
        for move_number, bucket in sorted(buckets.items())
        if bucket["count"] > 0
    ]


def aggregate_top1_accuracy_by_player_move(
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float]] = {}
    for row in eval_rows:
        target_rank = row.get("target_rank")
        ply_idx = row.get("ply_idx")
        if target_rank is None or ply_idx is None:
            continue
        move_number = int(
            row.get("player_move_number")
            or player_move_number(str(row.get("side") or ""), int(ply_idx))
        )
        bucket = buckets.setdefault(move_number, {"correct": 0.0, "count": 0.0})
        bucket["correct"] += 1.0 if int(target_rank) <= 1 else 0.0
        bucket["count"] += 1.0
    return [
        {
            "player_move_number": move_number,
            "top1_accuracy": bucket["correct"] / bucket["count"],
            "correct": int(bucket["correct"]),
            "count": int(bucket["count"]),
        }
        for move_number, bucket in sorted(buckets.items())
        if bucket["count"] > 0
    ]


def aggregate_top1_accuracy_comparison_by_player_move(
    successful_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_rows = collect_eval_rows_from_successful_players(successful_rows, "baseline_eval_rows.json")
    finetuned_rows = collect_eval_rows_from_successful_players(successful_rows, "finetuned_eval_rows.json")
    if not baseline_rows or not finetuned_rows:
        return []

    baseline_by_move = {
        row["player_move_number"]: row
        for row in aggregate_top1_accuracy_by_player_move(baseline_rows)
    }
    finetuned_by_move = {
        row["player_move_number"]: row
        for row in aggregate_top1_accuracy_by_player_move(finetuned_rows)
    }

    comparison_rows = []
    for move_number in sorted(set(baseline_by_move) & set(finetuned_by_move)):
        baseline_accuracy = float(baseline_by_move[move_number]["top1_accuracy"])
        finetuned_accuracy = float(finetuned_by_move[move_number]["top1_accuracy"])
        comparison_rows.append(
            {
                "player_move_number": move_number,
                "baseline_top1_accuracy": baseline_accuracy,
                "finetuned_top1_accuracy": finetuned_accuracy,
                "delta_top1_accuracy": finetuned_accuracy - baseline_accuracy,
                "baseline_count": baseline_by_move[move_number]["count"],
                "finetuned_count": finetuned_by_move[move_number]["count"],
            }
        )
    return comparison_rows


def plot_phase_target_probability_baseline(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    artifact_name: str = "baseline_eval_rows.json",
    output_filename: str = "phase_probability_baseline.png",
    zero_based_player_move_number: bool = False,
) -> Path | None:
    eval_rows = collect_eval_rows_from_successful_players(successful_rows, artifact_name)
    if not eval_rows:
        return None
    aggregated_rows = aggregate_target_probability_by_player_move(
        eval_rows,
        zero_based_player_move_number=zero_based_player_move_number,
    )
    if not aggregated_rows:
        return None

    x_values = [row["player_move_number"] for row in aggregated_rows]
    y_values = [row["average_target_probability"] for row in aggregated_rows]
    plt.figure(figsize=(11, 6))
    plt.plot(x_values, y_values, color="tab:blue", linewidth=2.2)
    plt.xlabel("Player move index" if zero_based_player_move_number else "Player move number")
    plt.ylabel("Average target probability")
    plt.title("Baseline Target Probability by Player Move Number")
    plt.grid(alpha=0.3)
    plot_path = root_dirs["plots"] / output_filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_phase_target_probability_finetuned(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    artifact_name: str = "finetuned_eval_rows.json",
    output_filename: str = "phase_probability_finetuned.png",
    zero_based_player_move_number: bool = False,
) -> Path | None:
    eval_rows = collect_eval_rows_from_successful_players(successful_rows, artifact_name)
    if not eval_rows:
        return None
    aggregated_rows = aggregate_target_probability_by_player_move(
        eval_rows,
        zero_based_player_move_number=zero_based_player_move_number,
    )
    if not aggregated_rows:
        return None

    x_values = [row["player_move_number"] for row in aggregated_rows]
    y_values = [row["average_target_probability"] for row in aggregated_rows]
    plt.figure(figsize=(11, 6))
    plt.plot(x_values, y_values, color="tab:green", linewidth=2.2)
    plt.xlabel("Player move index" if zero_based_player_move_number else "Player move number")
    plt.ylabel("Average target probability")
    plt.title("Fine-Tuned Target Probability by Player Move Number")
    plt.grid(alpha=0.3)
    plot_path = root_dirs["plots"] / output_filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_phase_target_probability_delta(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    baseline_artifact_name: str = "baseline_eval_rows.json",
    finetuned_artifact_name: str = "finetuned_eval_rows.json",
    output_filename: str = "phase_probability_delta.png",
    zero_based_player_move_number: bool = False,
) -> Path | None:
    aggregated_rows = aggregate_delta_target_probability_by_player_move(
        successful_rows,
        baseline_artifact_name=baseline_artifact_name,
        finetuned_artifact_name=finetuned_artifact_name,
        zero_based_player_move_number=zero_based_player_move_number,
    )
    if not aggregated_rows:
        return None

    x_values = [row["player_move_number"] for row in aggregated_rows]
    y_values = [row["average_delta_target_probability"] for row in aggregated_rows]
    plt.figure(figsize=(11, 6))
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.plot(x_values, y_values, color="tab:orange", linewidth=2.2)
    plt.xlabel("Player move index" if zero_based_player_move_number else "Player move number")
    plt.ylabel("Average delta target probability")
    plt.title("Fine-Tuned Minus Baseline Target Probability by Player Move Number")
    plt.grid(alpha=0.3)
    plot_path = root_dirs["plots"] / output_filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def plot_phase_top1_accuracy_comparison(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
) -> Path | None:
    aggregated_rows = aggregate_top1_accuracy_comparison_by_player_move(successful_rows)
    if not aggregated_rows:
        return None

    write_json(root_dirs["root"] / "phase_top1_accuracy_by_move.json", aggregated_rows)

    x_values = [row["player_move_number"] for row in aggregated_rows]
    baseline_values = [row["baseline_top1_accuracy"] for row in aggregated_rows]
    finetuned_values = [row["finetuned_top1_accuracy"] for row in aggregated_rows]
    delta_values = [row["delta_top1_accuracy"] for row in aggregated_rows]

    fig, accuracy_axis = plt.subplots(figsize=(12, 6.5))
    delta_axis = accuracy_axis.twinx()

    accuracy_axis.plot(
        x_values,
        baseline_values,
        color="tab:blue",
        linewidth=2.1,
        label="Baseline top-1",
    )
    accuracy_axis.plot(
        x_values,
        finetuned_values,
        color="tab:green",
        linewidth=2.1,
        label="Fine-tuned top-1",
    )
    delta_axis.axhline(0.0, color="tab:gray", linewidth=1.0, alpha=0.6)
    delta_axis.plot(
        x_values,
        delta_values,
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label="Delta top-1",
    )

    accuracy_axis.set_xlabel("Player move number")
    accuracy_axis.set_ylabel("Top-1 accuracy")
    delta_axis.set_ylabel("Delta top-1 accuracy")
    accuracy_axis.set_title("Top-1 Accuracy by Player Move Number")
    accuracy_axis.grid(alpha=0.3)

    lines = accuracy_axis.get_lines() + delta_axis.get_lines()
    labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
    visible_lines = [line for line in lines if not line.get_label().startswith("_")]
    accuracy_axis.legend(visible_lines, labels, loc="best")

    plot_path = root_dirs["plots"] / "phase_top1_accuracy_by_move.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def build_result_plots(
    root_dirs: dict[str, Path],
    successful_rows: list[dict[str, Any]],
    perf_type: str,
    training_loss_smoothing_window: int = 5,
    training_loss_grid_points: int = 200,
) -> dict[str, str]:
    return {
        "elo_vs_top1_accuracy": str(plot_elo_vs_top1_accuracy(root_dirs, successful_rows, perf_type) or ""),
        "elo_vs_delta_top1_accuracy": str(plot_elo_vs_delta_top1_accuracy(root_dirs, successful_rows, perf_type) or ""),
        "players_sorted_by_elo_top1_accuracy": str(plot_players_sorted_by_elo_top1_accuracy(root_dirs, successful_rows) or ""),
        "elo_vs_kl_divergence": str(plot_elo_vs_kl_divergence(root_dirs, successful_rows, perf_type) or ""),
        "training_loss_by_player": str(
            plot_training_loss_by_player(
                root_dirs,
                successful_rows,
                smoothing_window=training_loss_smoothing_window,
            )
            or ""
        ),
        "training_loss_mean": str(
            plot_mean_training_loss(
                root_dirs,
                successful_rows,
                smoothing_window=training_loss_smoothing_window,
                grid_points=training_loss_grid_points,
            )
            or ""
        ),
        "training_loss_quality_100_games": str(
            plot_mean_training_loss_quality(
                root_dirs,
                successful_rows,
                interval_games=100,
                smoothing_window=training_loss_smoothing_window,
                grid_points=max(400, training_loss_grid_points),
            )
            or ""
        ),
        "phase_probability_baseline": str(
            plot_phase_target_probability_baseline(root_dirs, successful_rows) or ""
        ),
        "phase_probability_finetuned": str(
            plot_phase_target_probability_finetuned(root_dirs, successful_rows) or ""
        ),
        "phase_probability_delta": str(
            plot_phase_target_probability_delta(root_dirs, successful_rows) or ""
        ),
        "phase_top1_accuracy_by_move": str(
            plot_phase_top1_accuracy_comparison(root_dirs, successful_rows) or ""
        ),
    }


def build_successful_row_from_final_report(
    player_dir: Path,
    final_report: dict[str, Any],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    profile = final_report.get("profile") or {}
    dataset_summary = final_report.get("dataset_summary") or {}
    baseline_metrics = final_report.get("baseline_test_metrics") or {}
    finetuned_metrics = final_report.get("finetuned_test_metrics") or {}
    comparison = final_report.get("comparison") or {}

    username = (
        final_report.get("username")
        or profile.get("username")
        or dataset_summary.get("username")
        or player_dir.name
    )
    if not dataset_summary:
        raise ValueError(f"{username}: final_report.json is missing dataset_summary.")
    if not baseline_metrics:
        raise ValueError(f"{username}: final_report.json is missing baseline_test_metrics.")
    if not finetuned_metrics:
        raise ValueError(f"{username}: final_report.json is missing finetuned_test_metrics.")

    row = {
        "username": username,
        "elo": profile.get("elo", dataset_summary.get("elo")),
        "profile_games": profile.get("games"),
        "parsed_games_used": dataset_summary.get("parsed_games_used"),
        "train_games_used": dataset_summary.get("train_games_used"),
        "val_games": dataset_summary.get("val_games"),
        "test_games": dataset_summary.get("test_games"),
        "train_examples": dataset_summary.get("train_examples"),
        "test_examples": dataset_summary.get("test_examples"),
        "baseline_mean_rank": baseline_metrics.get("mean_rank"),
        "finetuned_mean_rank": finetuned_metrics.get("mean_rank"),
        "delta_mean_rank": comparison.get(
            "delta_mean_rank",
            (finetuned_metrics.get("mean_rank") or 0.0) - (baseline_metrics.get("mean_rank") or 0.0),
        ),
        "average_kl_finetuned_vs_baseline": comparison.get("average_kl_finetuned_vs_baseline", 0.0),
        "player_results_dir": str(player_dir),
    }

    for top_k in top_ks:
        baseline_accuracy_key = f"top{top_k}_accuracy"
        baseline_correct_key = f"top{top_k}_correct"
        finetuned_accuracy_key = f"top{top_k}_accuracy"
        finetuned_correct_key = f"top{top_k}_correct"
        delta_key = f"delta_top{top_k}_accuracy"

        if baseline_accuracy_key not in baseline_metrics:
            raise ValueError(f"{username}: missing baseline metric {baseline_accuracy_key}.")
        if baseline_correct_key not in baseline_metrics:
            raise ValueError(f"{username}: missing baseline metric {baseline_correct_key}.")
        if finetuned_accuracy_key not in finetuned_metrics:
            raise ValueError(f"{username}: missing finetuned metric {finetuned_accuracy_key}.")
        if finetuned_correct_key not in finetuned_metrics:
            raise ValueError(f"{username}: missing finetuned metric {finetuned_correct_key}.")

        row[f"baseline_top{top_k}_accuracy"] = baseline_metrics[baseline_accuracy_key]
        row[f"baseline_top{top_k}_correct"] = baseline_metrics[baseline_correct_key]
        row[f"finetuned_top{top_k}_accuracy"] = finetuned_metrics[finetuned_accuracy_key]
        row[f"finetuned_top{top_k}_correct"] = finetuned_metrics[finetuned_correct_key]
        row[f"delta_top{top_k}_accuracy"] = comparison.get(
            delta_key,
            finetuned_metrics[finetuned_accuracy_key] - baseline_metrics[baseline_accuracy_key],
        )

    return row


def load_existing_train_games_comparison(
    results_dir: str | Path,
    training_loss_smoothing_window: int = 5,
    training_loss_grid_points: int = 200,
) -> dict[str, Any]:
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {root}")

    config_payload = read_json(config_path)
    top_ks = tuple(int(top_k) for top_k in (config_payload.get("top_ks") or (1, 3, 5, 10)))
    perf_type = str(config_payload.get("perf_type") or "classical")
    root_dirs = {
        "root": root,
        "players": root / "players",
        "plots": root / "plots",
    }
    root_dirs["plots"].mkdir(parents=True, exist_ok=True)

    successful_rows = []
    failed_rows = []
    player_dirs = sorted(
        [path for path in root_dirs["players"].iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )

    for player_dir in player_dirs:
        final_report_path = player_dir / "final_report.json"
        error_path = player_dir / "error.json"
        dataset_summary_path = player_dir / "dataset_summary.json"
        username_hint = player_dir.name
        if dataset_summary_path.exists():
            username_hint = (read_json(dataset_summary_path) or {}).get("username") or username_hint

        try:
            if final_report_path.exists():
                final_report = read_json(final_report_path)
                successful_rows.append(
                    build_successful_row_from_final_report(
                        player_dir=player_dir,
                        final_report=final_report,
                        top_ks=top_ks,
                    )
                )
                continue

            if error_path.exists():
                error_payload = read_json(error_path)
                failed_rows.append(
                    {
                        "username": error_payload.get("username") or username_hint,
                        "error_type": error_payload.get("error_type") or "PlayerRunFailed",
                        "error_message": error_payload.get("error_message") or "Unknown player failure.",
                        "traceback": error_payload.get("traceback") or "",
                    }
                )
                continue

            failed_rows.append(
                {
                    "username": username_hint,
                    "error_type": "IncompletePlayerArtifacts",
                    "error_message": (
                        f"Missing final_report.json in {player_dir}. "
                        "Fine-tuned evaluation metrics cannot be reconstructed from the remaining files alone."
                    ),
                    "traceback": "",
                }
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "username": username_hint,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    successful_rows = sorted(
        successful_rows,
        key=lambda row: (row["elo"] is None, row["elo"] if row["elo"] is not None else 10**9, row["username"].lower()),
    )
    aggregate_metrics = build_aggregate_metrics(successful_rows, top_ks)
    plots = build_result_plots(
        root_dirs=root_dirs,
        successful_rows=successful_rows,
        perf_type=perf_type,
        training_loss_smoothing_window=training_loss_smoothing_window,
        training_loss_grid_points=training_loss_grid_points,
    )

    comparison_summary = {
        "run_id": root.name,
        "config": config_payload,
        "aggregate_metrics": aggregate_metrics,
        "successful_players": successful_rows,
        "failed_players": failed_rows,
        "plots": plots,
    }
    write_json(root / "comparison_summary.json", comparison_summary)
    write_json(root / "aggregate_metrics.json", aggregate_metrics)

    if successful_rows:
        csv_path = root / "comparison_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(successful_rows[0].keys()))
            writer.writeheader()
            writer.writerows(successful_rows)

    return {
        "run_id": root.name,
        "results_dir": str(root),
        "comparison_summary": comparison_summary,
        "successful_rows": successful_rows,
        "failed_rows": failed_rows,
        "aggregate_metrics": aggregate_metrics,
    }


def generate_posthoc_result_plots(
    results_dir: str | Path,
    training_loss_smoothing_window: int = 5,
    training_loss_grid_points: int = 200,
) -> dict[str, str]:
    result = load_existing_train_games_comparison(
        results_dir=results_dir,
        training_loss_smoothing_window=training_loss_smoothing_window,
        training_loss_grid_points=training_loss_grid_points,
    )
    return (result.get("comparison_summary") or {}).get("plots") or {}


def generate_posthoc_phase_probability_plots(
    results_dir: str | Path,
    recompute_missing_eval_rows: bool = False,
    phase_min_context_ply: int | None = None,
    overwrite_phase_eval_rows: bool = False,
    zero_based_player_move_number: bool = False,
    training_loss_smoothing_window: int = 5,
    training_loss_grid_points: int = 200,
) -> dict[str, str]:
    result = load_existing_train_games_comparison(
        results_dir=results_dir,
        training_loss_smoothing_window=training_loss_smoothing_window,
        training_loss_grid_points=training_loss_grid_points,
    )
    if not recompute_missing_eval_rows and phase_min_context_ply is None:
        return (result.get("comparison_summary") or {}).get("plots") or {}

    root = Path(results_dir)
    config_payload = read_json(root / "config.json")
    successful_rows = result.get("successful_rows") or []
    if not successful_rows:
        return (result.get("comparison_summary") or {}).get("plots") or {}

    config_min_context_ply = int(config_payload.get("min_context_ply") or 0)
    eval_min_context_ply = config_min_context_ply if phase_min_context_ply is None else int(phase_min_context_ply)
    use_custom_phase_artifacts = (
        phase_min_context_ply is not None
        and eval_min_context_ply != config_min_context_ply
    )
    if use_custom_phase_artifacts:
        baseline_artifact_name = f"baseline_phase_eval_rows_min_context_ply_{eval_min_context_ply}.json"
        finetuned_artifact_name = f"finetuned_phase_eval_rows_min_context_ply_{eval_min_context_ply}.json"
    else:
        baseline_artifact_name = "baseline_eval_rows.json"
        finetuned_artifact_name = "finetuned_eval_rows.json"

    needs_eval_rows = []
    for row in successful_rows:
        player_dir = Path(row["player_results_dir"])
        baseline_eval_path = player_dir / baseline_artifact_name
        finetuned_eval_path = player_dir / finetuned_artifact_name
        if overwrite_phase_eval_rows or not (baseline_eval_path.exists() and finetuned_eval_path.exists()):
            needs_eval_rows.append(row)

    if needs_eval_rows:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = maybe_login_hf()
        tokenizer = load_tokenizer(str(config_payload.get("model_id") or ExperimentConfig.model_id), hf_token)
    else:
        device = ""
        hf_token = None
        tokenizer = None

    for row in needs_eval_rows:
        player_dir = Path(row["player_results_dir"])
        baseline_eval_path = player_dir / baseline_artifact_name
        finetuned_eval_path = player_dir / finetuned_artifact_name

        username = str(row["username"])
        print(f"Recomputing phase-eval rows for {username} with min_context_ply={eval_min_context_ply}...")
        raw_games = load_lichess_games_san(
            username=username,
            max_games=int(config_payload.get("max_games") or 0),
            perf_type=str(config_payload.get("perf_type") or "classical"),
            rated_only=bool(config_payload.get("rated_only")),
        )
        parsed_games = parse_target_games(raw_games, username)
        train_games, val_games, test_games = split_games_train_val_test(
            game_rows=parsed_games,
            split_seed=int(config_payload.get("split_seed") or 42),
            split_strategy=str(config_payload.get("split_strategy") or "chronological"),
            test_frac=float(config_payload.get("test_frac") or 0.2),
            val_frac_within_train=float(config_payload.get("val_frac_within_train") or 0.2),
        )
        test_examples = build_examples_from_games(
            test_games,
            eval_min_context_ply,
        )

        if not use_custom_phase_artifacts and row.get("parsed_games_used") and int(row["parsed_games_used"]) != len(parsed_games):
            print(
                f"{username}: parsed_games_used changed from {row['parsed_games_used']} to {len(parsed_games)} "
                "while rebuilding phase plots."
            )
        if not use_custom_phase_artifacts and row.get("test_examples") and int(row["test_examples"]) != len(test_examples):
            print(
                f"{username}: test_examples changed from {row['test_examples']} to {len(test_examples)} "
                "while rebuilding phase plots."
            )

        baseline_model = load_base_model(
            str(config_payload.get("model_id") or ExperimentConfig.model_id),
            tokenizer,
            device,
            hf_token,
        )
        _, _, baseline_eval_rows = evaluate_policy_model(
            model=baseline_model,
            tokenizer=tokenizer,
            examples=test_examples,
            device=device,
            top_ks=tuple(int(top_k) for top_k in (config_payload.get("top_ks") or (1, 3, 5, 10))),
            debug_n=int(config_payload.get("debug_examples") or 10),
            candidate_scoring_batch_size=int(config_payload.get("candidate_scoring_batch_size") or 64),
            max_length=int(config_payload.get("max_length") or 256),
            return_eval_rows=True,
        )
        write_json(baseline_eval_path, baseline_eval_rows or [])
        cleanup_torch_objects(baseline_model)
        del baseline_model

        finetuned_model = load_saved_player_model(player_dir, tokenizer, device)
        _, _, finetuned_eval_rows = evaluate_policy_model(
            model=finetuned_model,
            tokenizer=tokenizer,
            examples=test_examples,
            device=device,
            top_ks=tuple(int(top_k) for top_k in (config_payload.get("top_ks") or (1, 3, 5, 10))),
            debug_n=int(config_payload.get("debug_examples") or 10),
            candidate_scoring_batch_size=int(config_payload.get("candidate_scoring_batch_size") or 64),
            max_length=int(config_payload.get("max_length") or 256),
            return_eval_rows=True,
        )
        write_json(finetuned_eval_path, finetuned_eval_rows or [])
        cleanup_torch_objects(finetuned_model)
        del finetuned_model

    cleanup_torch_objects()
    refreshed = load_existing_train_games_comparison(
        results_dir=results_dir,
        training_loss_smoothing_window=training_loss_smoothing_window,
        training_loss_grid_points=training_loss_grid_points,
    )
    plots = dict((refreshed.get("comparison_summary") or {}).get("plots") or {})

    if use_custom_phase_artifacts or zero_based_player_move_number:
        root_dirs = root_dirs_from_existing_results(root)
        plots["phase_probability_baseline"] = str(
            plot_phase_target_probability_baseline(
                root_dirs,
                successful_rows,
                artifact_name=baseline_artifact_name,
                zero_based_player_move_number=zero_based_player_move_number,
            )
            or ""
        )
        plots["phase_probability_finetuned"] = str(
            plot_phase_target_probability_finetuned(
                root_dirs,
                successful_rows,
                artifact_name=finetuned_artifact_name,
                zero_based_player_move_number=zero_based_player_move_number,
            )
            or ""
        )
        plots["phase_probability_delta"] = str(
            plot_phase_target_probability_delta(
                root_dirs,
                successful_rows,
                baseline_artifact_name=baseline_artifact_name,
                finetuned_artifact_name=finetuned_artifact_name,
                zero_based_player_move_number=zero_based_player_move_number,
            )
            or ""
        )

        comparison_summary_path = root / "comparison_summary.json"
        if comparison_summary_path.exists():
            comparison_summary = read_json(comparison_summary_path)
            comparison_summary["plots"] = {
                **(comparison_summary.get("plots") or {}),
                **plots,
            }
            comparison_summary["phase_probability_plot_context"] = {
                "metric_min_context_ply": config_min_context_ply,
                "plot_min_context_ply": eval_min_context_ply,
                "zero_based_player_move_number": zero_based_player_move_number,
                "baseline_artifact_name": baseline_artifact_name,
                "finetuned_artifact_name": finetuned_artifact_name,
            }
            write_json(comparison_summary_path, comparison_summary)

    return plots


def plot_train_games_learning_curve(
    root_dirs: dict[str, Path],
    curve_rows: list[dict[str, Any]],
    username: str,
) -> Path | None:
    if not curve_rows:
        return None

    rows = sorted(curve_rows, key=lambda row: int(row["train_games_used"]))
    x_values = [int(row["train_games_used"]) for row in rows]
    finetuned_top1 = [float(row["finetuned_top1_accuracy"]) * 100.0 for row in rows]
    baseline_top1 = [float(row["baseline_top1_accuracy"]) * 100.0 for row in rows]

    fig, accuracy_axis = plt.subplots(figsize=(11, 6.5))
    accuracy_axis.plot(x_values, finetuned_top1, marker="o", linewidth=2.4, label="Finetuned top-1")
    accuracy_axis.plot(x_values, baseline_top1, linestyle="--", linewidth=1.8, label="Baseline top-1")
    accuracy_axis.set_xlabel("Training games")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_title("Learning curve")
    accuracy_axis.set_xlim(left=0, right=max(x_values) * 1.03)
    accuracy_axis.xaxis.set_major_locator(MultipleLocator(200))
    accuracy_axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    accuracy_axis.tick_params(axis="x", labelrotation=45)
    accuracy_axis.grid(True, which="major", axis="both", alpha=0.25)

    accuracy_axis.legend(loc="best")
    fig.tight_layout()

    plot_path = root_dirs["plots"] / "train_games_learning_curve.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def run_train_games_learning_curve(
    config: ExperimentConfig | None = None,
    username: str | None = None,
    resume_results_dir: str | Path | None = None,
    skip_completed: bool = True,
) -> dict[str, Any]:
    if resume_results_dir is not None:
        root_dirs = root_dirs_from_existing_results(resume_results_dir)
        run_id = root_dirs["root"].name
        config_path = root_dirs["root"] / "config.json"
        if config is None:
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config.json in resume directory: {root_dirs['root']}")
            config = experiment_config_from_payload(read_json(config_path))
        write_json(config_path, asdict(config))
    else:
        config = config or ExperimentConfig()
        run_id = make_run_id()
        root_dirs = ensure_root_dirs(run_id, config.results_root_name)
        write_json(root_dirs["root"] / "config.json", asdict(config))

    curve_game_counts = tuple(int(value) for value in config.learning_curve_train_games if int(value) > 0)
    if not curve_game_counts:
        raise ValueError("Set config.learning_curve_train_games, for example (500, 1000, 2000, 5000, 10000, 20000).")

    selected_username = username or config.learning_curve_player_username
    if not selected_username:
        if not config.player_usernames:
            raise ValueError("Provide username or set config.learning_curve_player_username.")
        selected_username = config.player_usernames[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = maybe_login_hf()
    seed_everything(config.split_seed)

    print("Run ID:", run_id)
    print("Learning curve player:", selected_username)
    print("Train game counts:", curve_game_counts)
    print("Device:", device)
    print("Results root:", root_dirs["root"])

    tokenizer = load_tokenizer(config.model_id, hf_token)
    player_dirs = ensure_player_dirs(root_dirs, selected_username)
    curve_root = player_dirs["root"] / "learning_curve"
    curve_root.mkdir(parents=True, exist_ok=True)

    write_player_status(player_dirs, selected_username, "learning_curve_loading_data", run_id=run_id)
    profile = load_lichess_user_profile(selected_username, config.perf_type)
    raw_games = load_lichess_games_san(
        username=selected_username,
        max_games=config.max_games,
        perf_type=config.perf_type,
        rated_only=config.rated_only,
    )
    parsed_games = parse_target_games(raw_games, selected_username)
    min_required_games = min_total_games_required(config)
    if len(parsed_games) < min_required_games:
        raise ValueError(
            f"{selected_username} has only {len(parsed_games)} parsed games, but at least "
            f"{min_required_games} are required to create train, validation, and test splits."
        )

    train_games, val_games, test_games = split_games_train_val_test(
        parsed_games,
        split_seed=config.split_seed,
        split_strategy=config.split_strategy,
        test_frac=config.test_frac,
        val_frac_within_train=config.val_frac_within_train,
    )
    val_examples = build_examples_from_games(val_games, config.min_context_ply)
    test_examples = build_examples_from_games(test_games, config.min_context_ply)

    dataset_summary = {
        "username": selected_username,
        "elo": profile["elo"],
        "title": profile["title"],
        "perf_type": config.perf_type,
        "raw_games_loaded": len(raw_games),
        "parsed_games_used": len(parsed_games),
        "split_strategy": config.split_strategy,
        "train_games_available": len(train_games),
        "val_games": len(val_games),
        "test_games": len(test_games),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "learning_curve_train_games": list(curve_game_counts),
    }
    write_json(player_dirs["root"] / "dataset_summary.json", dataset_summary)

    baseline_metrics_path = curve_root / "baseline_metrics.json"
    baseline_distributions_path = curve_root / "baseline_distributions.json"
    baseline_eval_rows_path = curve_root / "baseline_eval_rows.json"
    if skip_completed and baseline_metrics_path.exists() and baseline_distributions_path.exists():
        baseline_test_metrics = read_json(baseline_metrics_path)
        baseline_test_distributions = read_json(baseline_distributions_path)
    else:
        write_player_status(player_dirs, selected_username, "learning_curve_evaluating_baseline", run_id=run_id)
        baseline_model = load_base_model(config.model_id, tokenizer, device, hf_token)
        baseline_test_metrics, baseline_test_distributions, baseline_eval_rows = evaluate_policy_model(
            model=baseline_model,
            tokenizer=tokenizer,
            examples=test_examples,
            device=device,
            top_ks=config.top_ks,
            debug_n=config.debug_examples,
            candidate_scoring_batch_size=config.candidate_scoring_batch_size,
            max_length=config.max_length,
            return_distributions=True,
            return_eval_rows=True,
        )
        write_json(baseline_metrics_path, baseline_test_metrics)
        write_json(baseline_distributions_path, baseline_test_distributions or [])
        write_json(baseline_eval_rows_path, baseline_eval_rows or [])
        print("baseline:", summarize_top_metrics(baseline_test_metrics, config.top_ks))
        cleanup_torch_objects(baseline_model)
        del baseline_model

    curve_rows = []
    failed_rows = []
    for game_count in sorted(set(curve_game_counts)):
        point_name = f"games_{game_count:05d}"
        point_root = curve_root / point_name
        point_dirs = {
            "root": point_root,
            "scratch": point_root / "scratch",
            "best_model": point_root / "best_model",
        }
        for path_obj in point_dirs.values():
            path_obj.mkdir(parents=True, exist_ok=True)

        final_report_path = point_root / "final_report.json"
        if skip_completed and final_report_path.exists():
            print(f"Skipping completed curve point: {game_count} games")
            curve_rows.append(read_json(final_report_path)["summary_row"])
            continue

        try:
            selected_train_games = select_fixed_train_games(
                train_games,
                train_games_per_player=game_count,
                split_strategy=config.split_strategy,
                split_seed=config.split_seed,
                username=selected_username,
                require_train_games_per_player=config.require_train_games_per_player,
            )
            train_examples = build_examples_from_games(selected_train_games, config.min_context_ply)
            point_dataset_summary = {
                **dataset_summary,
                "train_games_requested": game_count,
                "train_games_used": len(selected_train_games),
                "train_examples": len(train_examples),
            }
            write_json(point_root / "dataset_summary.json", point_dataset_summary)

            print("=" * 100)
            print(f"Learning curve point: {game_count} train games ({len(train_examples)} examples)")
            write_player_status(
                player_dirs,
                selected_username,
                "learning_curve_training_lora",
                run_id=run_id,
                train_games=game_count,
                train_examples=len(train_examples),
            )
            finetuned_model, train_summary = train_lora_model(
                config=config,
                tokenizer=tokenizer,
                train_examples=train_examples,
                model_id=config.model_id,
                player_dirs=point_dirs,
                run_name=point_name,
                device=device,
                hf_token=hf_token,
                username=selected_username,
                run_id=run_id,
            )
            write_json(point_root / "trainer_logs.json", train_summary)

            write_player_status(
                player_dirs,
                selected_username,
                "learning_curve_evaluating_finetuned",
                run_id=run_id,
                train_games=game_count,
                test_examples=len(test_examples),
            )
            finetuned_test_metrics, finetuned_test_distributions, finetuned_eval_rows = evaluate_policy_model(
                model=finetuned_model,
                tokenizer=tokenizer,
                examples=test_examples,
                device=device,
                top_ks=config.top_ks,
                debug_n=config.debug_examples,
                candidate_scoring_batch_size=config.candidate_scoring_batch_size,
                max_length=config.max_length,
                return_distributions=True,
                return_eval_rows=True,
            )
            write_json(point_root / "finetuned_eval_rows.json", finetuned_eval_rows or [])
            print("finetuned:", summarize_top_metrics(finetuned_test_metrics, config.top_ks))

            comparison = {
                f"delta_top{top_k}_accuracy": (
                    finetuned_test_metrics[f"top{top_k}_accuracy"] - baseline_test_metrics[f"top{top_k}_accuracy"]
                )
                for top_k in config.top_ks
            }
            comparison["delta_mean_rank"] = finetuned_test_metrics["mean_rank"] - baseline_test_metrics["mean_rank"]
            comparison["average_kl_finetuned_vs_baseline"] = average_kl_divergence(
                finetuned_test_distributions or [],
                baseline_test_distributions or [],
            )

            summary_row = {
                "username": selected_username,
                "elo": profile["elo"],
                "profile_games": profile["games"],
                "parsed_games_used": len(parsed_games),
                "train_games_requested": game_count,
                "train_games_used": len(selected_train_games),
                "val_games": len(val_games),
                "test_games": len(test_games),
                "train_examples": len(train_examples),
                "test_examples": len(test_examples),
                "baseline_mean_rank": baseline_test_metrics["mean_rank"],
                "finetuned_mean_rank": finetuned_test_metrics["mean_rank"],
                "delta_mean_rank": comparison["delta_mean_rank"],
                "average_kl_finetuned_vs_baseline": comparison["average_kl_finetuned_vs_baseline"],
                "point_results_dir": str(point_root),
            }
            for top_k in config.top_ks:
                summary_row[f"baseline_top{top_k}_accuracy"] = baseline_test_metrics[f"top{top_k}_accuracy"]
                summary_row[f"baseline_top{top_k}_correct"] = baseline_test_metrics[f"top{top_k}_correct"]
                summary_row[f"finetuned_top{top_k}_accuracy"] = finetuned_test_metrics[f"top{top_k}_accuracy"]
                summary_row[f"finetuned_top{top_k}_correct"] = finetuned_test_metrics[f"top{top_k}_correct"]
                summary_row[f"delta_top{top_k}_accuracy"] = comparison[f"delta_top{top_k}_accuracy"]

            final_report = {
                "run_id": run_id,
                "username": selected_username,
                "model_id": config.model_id,
                "profile": profile,
                "config": asdict(config),
                "dataset_summary": point_dataset_summary,
                "baseline_test_metrics": baseline_test_metrics,
                "finetuned_test_metrics": finetuned_test_metrics,
                "comparison": comparison,
                "training": train_summary,
                "summary_row": summary_row,
            }
            write_json(final_report_path, final_report)
            if config.learning_curve_save_models:
                save_player_model(
                    config=config,
                    model=finetuned_model,
                    tokenizer=tokenizer,
                    player_dirs=point_dirs,
                    player_summary=final_report,
                )

            curve_rows.append(summary_row)
            cleanup_torch_objects(finetuned_model)
            del finetuned_model

        except (KeyboardInterrupt, SystemExit):
            cleanup_torch_objects()
            raise
        except Exception as exc:
            error_payload = {
                "username": selected_username,
                "train_games_requested": game_count,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed_rows.append(error_payload)
            write_json(point_root / "error.json", error_payload)
            cleanup_torch_objects()
            print(f"FAILED for {selected_username} at {game_count} games: {type(exc).__name__}: {exc}")

    curve_rows = sorted(curve_rows, key=lambda row: int(row["train_games_used"]))
    plot_path = plot_train_games_learning_curve(root_dirs, curve_rows, selected_username)
    summary = {
        "run_id": run_id,
        "username": selected_username,
        "config": asdict(config),
        "dataset_summary": dataset_summary,
        "baseline_test_metrics": baseline_test_metrics,
        "curve_points": curve_rows,
        "failed_points": failed_rows,
        "plots": {
            "train_games_learning_curve": str(plot_path or ""),
        },
    }
    write_json(root_dirs["root"] / "learning_curve_summary.json", summary)

    if curve_rows:
        csv_path = root_dirs["root"] / "learning_curve_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curve_rows)

    write_player_status(player_dirs, selected_username, "learning_curve_completed", run_id=run_id)
    print("Saved learning curve summary to:", root_dirs["root"] / "learning_curve_summary.json")
    if plot_path:
        print("Saved learning curve plot to:", plot_path)

    return {
        "run_id": run_id,
        "results_dir": str(root_dirs["root"]),
        "learning_curve_summary": summary,
        "curve_rows": curve_rows,
        "failed_rows": failed_rows,
        "plot_path": str(plot_path or ""),
    }


def run_fixed_train_games_comparison(
    config: ExperimentConfig | None = None,
    resume_results_dir: str | Path | None = None,
    skip_completed: bool = True,
) -> dict[str, Any]:
    if resume_results_dir is not None:
        root_dirs = root_dirs_from_existing_results(resume_results_dir)
        run_id = root_dirs["root"].name
        config_path = root_dirs["root"] / "config.json"
        if config is None:
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config.json in resume directory: {root_dirs['root']}")
            config = experiment_config_from_payload(read_json(config_path))
        write_json(config_path, asdict(config))
    else:
        config = config or ExperimentConfig()
        run_id = make_run_id()
        root_dirs = ensure_root_dirs(run_id, config.results_root_name)
        write_json(root_dirs["root"] / "config.json", asdict(config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = maybe_login_hf()
    seed_everything(config.split_seed)

    if not config.player_usernames:
        raise ValueError("Provide at least one player username.")

    print("Run ID:", run_id)
    print("Resume mode:", bool(resume_results_dir))
    print("Device:", device)
    print("Results root:", root_dirs["root"])
    print("Players:", config.player_usernames)
    print("Min parsed games required per player for split:", min_total_games_required(config))

    tokenizer = load_tokenizer(config.model_id, hf_token)
    successful_rows = []
    failed_rows = []

    for username in config.player_usernames:
        player_dirs = ensure_player_dirs(root_dirs, username)
        print("=" * 100)
        print(f"Processing player: {username}")
        write_player_status(player_dirs, username, "started", run_id=run_id)

        try:
            final_report_path = player_dirs["root"] / "final_report.json"
            if skip_completed and final_report_path.exists():
                print(f"Skipping completed player: {username}")
                write_player_status(player_dirs, username, "skipped_completed", run_id=run_id)
                continue
            error_path = player_dirs["root"] / "error.json"
            if error_path.exists():
                error_path.unlink()

            write_player_status(player_dirs, username, "loading_profile_and_games", run_id=run_id)
            profile = load_lichess_user_profile(username, config.perf_type)
            raw_games = load_lichess_games_san(
                username=username,
                max_games=config.max_games,
                perf_type=config.perf_type,
                rated_only=config.rated_only,
            )
            parsed_games = parse_target_games(raw_games, username)

            min_required_games = min_total_games_required(config)
            if len(parsed_games) < min_required_games:
                raise ValueError(
                    f"{username} has only {len(parsed_games)} parsed games, but at least {min_required_games} are required "
                    "to create train, validation, and test splits."
                )

            train_games, val_games, test_games = split_games_train_val_test(
                parsed_games,
                split_seed=config.split_seed,
                split_strategy=config.split_strategy,
                test_frac=config.test_frac,
                val_frac_within_train=config.val_frac_within_train,
            )
            selected_train_games = select_fixed_train_games(
                train_games,
                train_games_per_player=config.train_games_per_player,
                split_strategy=config.split_strategy,
                split_seed=config.split_seed,
                username=username,
                require_train_games_per_player=config.require_train_games_per_player,
            )

            train_examples = build_examples_from_games(selected_train_games, config.min_context_ply)
            val_examples = build_examples_from_games(val_games, config.min_context_ply)
            test_examples = build_examples_from_games(test_games, config.min_context_ply)

            player_dataset_summary = {
                "username": username,
                "elo": profile["elo"],
                "title": profile["title"],
                "perf_type": config.perf_type,
                "raw_games_loaded": len(raw_games),
                "parsed_games_used": len(parsed_games),
                "split_strategy": config.split_strategy,
                "train_games_available": len(train_games),
                "train_games_used": len(selected_train_games),
                "val_games": len(val_games),
                "test_games": len(test_games),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "test_examples": len(test_examples),
            }
            write_json(player_dirs["root"] / "dataset_summary.json", player_dataset_summary)

            write_player_status(
                player_dirs,
                username,
                "evaluating_baseline",
                run_id=run_id,
                train_examples=len(train_examples),
                test_examples=len(test_examples),
            )
            baseline_model = load_base_model(config.model_id, tokenizer, device, hf_token)
            baseline_test_metrics, baseline_test_distributions, baseline_eval_rows = evaluate_policy_model(
                model=baseline_model,
                tokenizer=tokenizer,
                examples=test_examples,
                device=device,
                top_ks=config.top_ks,
                debug_n=config.debug_examples,
                candidate_scoring_batch_size=config.candidate_scoring_batch_size,
                max_length=config.max_length,
                return_distributions=True,
                return_eval_rows=True,
            )
            write_json(player_dirs["root"] / "baseline_metrics.json", baseline_test_metrics)
            write_json(player_dirs["root"] / "baseline_eval_rows.json", baseline_eval_rows or [])
            print("baseline:", summarize_top_metrics(baseline_test_metrics, config.top_ks))
            cleanup_torch_objects(baseline_model)
            del baseline_model

            write_player_status(
                player_dirs,
                username,
                "training_lora",
                run_id=run_id,
                train_examples=len(train_examples),
                per_device_train_batch_size=config.per_device_train_batch_size,
                num_train_epochs=config.num_train_epochs,
            )
            finetuned_model, train_summary = train_lora_model(
                config=config,
                tokenizer=tokenizer,
                train_examples=train_examples,
                model_id=config.model_id,
                player_dirs=player_dirs,
                run_name="fixed_train_games",
                device=device,
                hf_token=hf_token,
                username=username,
                run_id=run_id,
            )
            write_json(player_dirs["root"] / "trainer_logs.json", train_summary)

            write_player_status(
                player_dirs,
                username,
                "evaluating_finetuned",
                run_id=run_id,
                test_examples=len(test_examples),
            )
            finetuned_test_metrics, finetuned_test_distributions, finetuned_eval_rows = evaluate_policy_model(
                model=finetuned_model,
                tokenizer=tokenizer,
                examples=test_examples,
                device=device,
                top_ks=config.top_ks,
                debug_n=config.debug_examples,
                candidate_scoring_batch_size=config.candidate_scoring_batch_size,
                max_length=config.max_length,
                return_distributions=True,
                return_eval_rows=True,
            )
            write_json(player_dirs["root"] / "finetuned_eval_rows.json", finetuned_eval_rows or [])
            print("finetuned:", summarize_top_metrics(finetuned_test_metrics, config.top_ks))

            comparison = {
                "delta_top1_accuracy": finetuned_test_metrics["top1_accuracy"] - baseline_test_metrics["top1_accuracy"],
                "delta_top3_accuracy": finetuned_test_metrics["top3_accuracy"] - baseline_test_metrics["top3_accuracy"],
                "delta_top5_accuracy": finetuned_test_metrics["top5_accuracy"] - baseline_test_metrics["top5_accuracy"],
                "delta_top10_accuracy": finetuned_test_metrics["top10_accuracy"] - baseline_test_metrics["top10_accuracy"],
                "delta_mean_rank": finetuned_test_metrics["mean_rank"] - baseline_test_metrics["mean_rank"],
                "average_kl_finetuned_vs_baseline": average_kl_divergence(
                    finetuned_test_distributions or [],
                    baseline_test_distributions or [],
                ),
            }

            final_report = {
                "run_id": run_id,
                "username": username,
                "model_id": config.model_id,
                "profile": profile,
                "config": asdict(config),
                "dataset_summary": player_dataset_summary,
                "baseline_test_metrics": baseline_test_metrics,
                "finetuned_test_metrics": finetuned_test_metrics,
                "comparison": comparison,
                "training": train_summary,
            }
            write_json(player_dirs["root"] / "final_report.json", final_report)
            save_player_model(
                config=config,
                model=finetuned_model,
                tokenizer=tokenizer,
                player_dirs=player_dirs,
                player_summary=final_report,
            )

            successful_rows.append(
                {
                    "username": username,
                    "elo": profile["elo"],
                    "profile_games": profile["games"],
                    "parsed_games_used": len(parsed_games),
                    "train_games_used": len(selected_train_games),
                    "val_games": len(val_games),
                    "test_games": len(test_games),
                    "train_examples": len(train_examples),
                    "test_examples": len(test_examples),
                    "baseline_top1_accuracy": baseline_test_metrics["top1_accuracy"],
                    "baseline_top1_correct": baseline_test_metrics["top1_correct"],
                    "baseline_top3_accuracy": baseline_test_metrics["top3_accuracy"],
                    "baseline_top3_correct": baseline_test_metrics["top3_correct"],
                    "baseline_top5_accuracy": baseline_test_metrics["top5_accuracy"],
                    "baseline_top5_correct": baseline_test_metrics["top5_correct"],
                    "baseline_top10_accuracy": baseline_test_metrics["top10_accuracy"],
                    "baseline_top10_correct": baseline_test_metrics["top10_correct"],
                    "baseline_mean_rank": baseline_test_metrics["mean_rank"],
                    "finetuned_top1_accuracy": finetuned_test_metrics["top1_accuracy"],
                    "finetuned_top1_correct": finetuned_test_metrics["top1_correct"],
                    "finetuned_top3_accuracy": finetuned_test_metrics["top3_accuracy"],
                    "finetuned_top3_correct": finetuned_test_metrics["top3_correct"],
                    "finetuned_top5_accuracy": finetuned_test_metrics["top5_accuracy"],
                    "finetuned_top5_correct": finetuned_test_metrics["top5_correct"],
                    "finetuned_top10_accuracy": finetuned_test_metrics["top10_accuracy"],
                    "finetuned_top10_correct": finetuned_test_metrics["top10_correct"],
                    "finetuned_mean_rank": finetuned_test_metrics["mean_rank"],
                    "delta_top1_accuracy": comparison["delta_top1_accuracy"],
                    "delta_top3_accuracy": comparison["delta_top3_accuracy"],
                    "delta_top5_accuracy": comparison["delta_top5_accuracy"],
                    "delta_top10_accuracy": comparison["delta_top10_accuracy"],
                    "delta_mean_rank": comparison["delta_mean_rank"],
                    "average_kl_finetuned_vs_baseline": comparison["average_kl_finetuned_vs_baseline"],
                    "player_results_dir": str(player_dirs["root"]),
                }
            )

            cleanup_torch_objects(finetuned_model)
            del finetuned_model
            write_player_status(player_dirs, username, "completed", run_id=run_id)

        except (KeyboardInterrupt, SystemExit) as exc:
            error_payload = {
                "username": username,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(player_dirs["root"] / "error.json", error_payload)
            write_player_status(
                player_dirs,
                username,
                "interrupted",
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            cleanup_torch_objects()
            raise
        except Exception as exc:
            error_payload = {
                "username": username,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed_rows.append(error_payload)
            write_json(player_dirs["root"] / "error.json", error_payload)
            write_player_status(
                player_dirs,
                username,
                "failed",
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            cleanup_torch_objects()
            print(f"FAILED for {username}: {type(exc).__name__}: {exc}")

    refreshed = load_existing_train_games_comparison(root_dirs["root"])
    successful_rows = refreshed["successful_rows"]
    failed_rows = refreshed["failed_rows"]
    aggregate_metrics = refreshed["aggregate_metrics"]
    comparison_summary = refreshed["comparison_summary"]

    print("Successful players:", len(successful_rows))
    print("Failed players:", len(failed_rows))
    print("Saved comparison summary to:", root_dirs["root"] / "comparison_summary.json")

    return {
        "run_id": run_id,
        "results_dir": str(root_dirs["root"]),
        "comparison_summary": comparison_summary,
        "successful_rows": successful_rows,
        "failed_rows": failed_rows,
        "aggregate_metrics": aggregate_metrics,
    }


def resume_fixed_train_games_comparison(
    results_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    return run_fixed_train_games_comparison(
        config=config,
        resume_results_dir=results_dir,
        skip_completed=True,
    )
