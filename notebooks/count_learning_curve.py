from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import matplotlib.pyplot as plt
import requests
import torch
from huggingface_hub import login
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


TokenizedRow = dict[str, list[int]]


@dataclass
class ExperimentConfig:
    username: str = "Vlad_Lazarev79"
    usernames: list[str] | None = None
    perf_type: str = "classical"
    max_games: int = 500
    rated_only: bool = False
    split_seed: int = 42
    split_strategy: str = "chronological"
    test_frac: float = 0.2
    val_frac_within_train: float = 0.2
    min_context_ply: int = 10
    count_step_games: int = 15
    min_train_games: int = 30
    max_train_games_for_curve: int | None = None
    multi_player_min_curve_games_for_aggregate: int | None = None
    scan_val_subset_examples: int = 1000
    candidate_top_k: int = 8
    candidate_neighbor_radius: int = 1
    candidate_relative_margin: float = 0.004
    sweet_spot_relative_headroom: float = 0.003
    sweet_spot_absolute_loss_delta: float = 0.005
    model_id: str = "daavidhauser/chess-bot-3000-250m"
    max_length: int = 256
    learning_rate: float = 2e-4
    num_train_epochs_per_block: int = 1
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    shuffle_new_block_examples: bool = True
    plot_smoothing_window: int = 5
    per_device_train_batch_size: int = 4 if torch.cuda.is_available() else 1
    per_device_eval_batch_size: int = 4 if torch.cuda.is_available() else 1
    logging_steps: int = 25


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


def ensure_results_dirs(
    config: ExperimentConfig,
    run_id: str,
    root_override: Path | None = None,
) -> dict[str, Path]:
    root = root_override or (Path("results") / f"{config.username}_{config.perf_type}_{run_id}")
    dirs = {
        "root": root,
        "curve_checkpoints": root / "curve_checkpoints",
        "best_model": root / "best_model",
        "plots": root / "plots",
    }
    for path_obj in dirs.values():
        path_obj.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_multi_results_dirs(
    config: ExperimentConfig,
    run_id: str,
    usernames: list[str],
) -> dict[str, Path]:
    root = Path("results") / f"multi_{len(usernames)}players_{config.perf_type}_{run_id}"
    dirs = {
        "root": root,
        "players": root / "players",
        "plots": root / "plots",
    }
    for path_obj in dirs.values():
        path_obj.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_usernames(config: ExperimentConfig) -> list[str]:
    raw_usernames = config.usernames if config.usernames else [config.username]
    usernames: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_usernames:
        name = (raw_name or "").strip()
        if not name:
            continue
        name_key = name.lower()
        if name_key in seen:
            continue
        usernames.append(name)
        seen.add(name_key)
    if not usernames:
        raise ValueError("Provide at least one non-empty username or usernames entry.")
    return usernames


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty numeric list.")
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


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
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def maybe_login_hf() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Hugging Face token detected.")
    else:
        print("No HF token found. Proceeding without explicit login.")
    return token


def load_lichess_games_san(
    username: str,
    max_games: int = 2000,
    perf_type: str = "classical",
    rated_only: bool = False,
) -> list[dict[str, Any]]:
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {"Accept": "application/x-ndjson"}
    params = {
        "max": max_games,
        "moves": "true",
        "pgnInJson": "false",
        "opening": "false",
        "clocks": "false",
        "evals": "false",
        "perfType": perf_type,
    }
    if rated_only:
        params["rated"] = "true"

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    raw_games = [json.loads(line) for line in response.text.splitlines() if line.strip()]
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
                    "uci_moves": uci_moves,
                }
            )
    return parsed_games


def split_games_train_val_test(
    game_rows: list[dict[str, Any]],
    split_seed: int = 42,
    split_strategy: str = "chronological",
    test_frac: float = 0.2,
    val_frac_within_train: float = 0.2,
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
                        "context": " ".join(moves[:ply_idx]),
                        "target": move_uci,
                    }
                )
            board.push(chess.Move.from_uci(move_uci))
    return examples


def order_train_games_for_curve(
    train_rows: list[dict[str, Any]],
    split_strategy: str,
    split_seed: int,
) -> list[dict[str, Any]]:
    rows = list(train_rows)
    if split_strategy == "chronological":
        return rows

    rng = random.Random(split_seed)
    rng.shuffle(rows)
    return rows


def select_eval_examples_subset(
    examples: list[dict[str, Any]],
    max_examples: int,
    seed: int,
    subset_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        raise ValueError("scan_val_subset_examples must be positive")
    if len(examples) <= max_examples:
        return list(examples), {
            "subset_name": subset_name,
            "selection": "full",
            "total_examples": len(examples),
            "selected_examples": len(examples),
            "seed": seed,
        }

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:max_examples])
    subset = [examples[idx] for idx in selected_indices]
    return subset, {
        "subset_name": subset_name,
        "selection": "fixed_random_subset",
        "total_examples": len(examples),
        "selected_examples": len(subset),
        "seed": seed,
    }


def build_counts_to_scan(min_curve_games: int, max_curve_games: int, step_games: int) -> list[int]:
    counts = list(range(min_curve_games, max_curve_games + 1, step_games))
    if counts and counts[-1] != max_curve_games:
        counts.append(max_curve_games)
    if not counts:
        raise ValueError("No train-game counts available for the curve.")
    return counts


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


def load_lora_checkpoint_model(
    model_id: str,
    tokenizer: AutoTokenizer,
    checkpoint_dir: Path,
    device: str,
    token: str | None,
) -> AutoModelForCausalLM:
    base_model = load_base_model(model_id, tokenizer, device, token)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    model.to(device)
    model.eval()
    return model


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
    base_model = load_base_model(model_id, tokenizer, device, token)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
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


def tokenize_for_lm(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> TokenizedRow:
    prompt_ids = get_prompt_ids(tokenizer, example["context"])
    target_ids = tokenizer(" " + example["target"], add_special_tokens=False)["input_ids"]
    max_prompt_len = max(0, max_length - len(target_ids))
    trimmed_prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
    input_ids = trimmed_prompt_ids + target_ids
    labels = ([-100] * len(trimmed_prompt_ids)) + target_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def prepare_tokenized_rows(
    examples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> tuple[list[TokenizedRow], dict[str, Any]]:
    tokenized_rows = [tokenize_for_lm(example, tokenizer, max_length) for example in examples]
    lengths = [len(row["input_ids"]) for row in tokenized_rows]
    tokenization_summary = {
        "num_rows": len(tokenized_rows),
        "max_length": max_length,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "avg_tokens": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    return tokenized_rows, tokenization_summary


def make_causal_lm_data_collator(tokenizer: AutoTokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collator(features: list[TokenizedRow]) -> dict[str, torch.Tensor]:
        max_seq_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_seq_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + ([pad_token_id] * pad_len))
            batch["attention_mask"].append(feature["attention_mask"] + ([0] * pad_len))
            batch["labels"].append(feature["labels"] + ([-100] * pad_len))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collator


def make_shuffle_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def evaluate_lm_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_rows: list[TokenizedRow],
    batch_size: int,
    device: str,
    run_name: str,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    dataloader = DataLoader(
        eval_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_causal_lm_data_collator(tokenizer),
        pin_memory=torch.cuda.is_available(),
    )

    total_weighted_loss = 0.0
    total_valid_tokens = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(**batch)
            batch_loss = float(outputs.loss.detach().cpu().item())
            valid_tokens = int((batch["labels"] != -100).sum().detach().cpu().item())
            if valid_tokens <= 0:
                continue
            total_weighted_loss += batch_loss * valid_tokens
            total_valid_tokens += valid_tokens
            total_batches += 1

    if was_training:
        model.train()

    if total_valid_tokens <= 0:
        raise ValueError(f"No valid label tokens found during evaluation for {run_name}")

    eval_loss = total_weighted_loss / total_valid_tokens
    perplexity = float(math.exp(min(eval_loss, 20.0)))
    metrics = {
        "eval_loss": eval_loss,
        "eval_perplexity": perplexity,
        "num_eval_examples": len(eval_rows),
        "num_eval_tokens": total_valid_tokens,
        "num_eval_batches": total_batches,
    }
    return {
        "loss": float(eval_loss),
        "perplexity": perplexity,
        "raw_metrics": to_builtin(metrics),
    }


def train_incremental_block(
    model: AutoModelForCausalLM,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None,
    tokenizer: AutoTokenizer,
    train_rows: list[TokenizedRow],
    config: ExperimentConfig,
    device: str,
    run_name: str,
    global_step_start: int,
    shuffle_seed: int,
) -> tuple[int, dict[str, Any]]:
    if not train_rows:
        raise ValueError(f"No training examples were generated for {run_name}")

    tokenization_summary = {
        "num_rows": len(train_rows),
        "max_length": config.max_length,
        "min_tokens": min(len(row["input_ids"]) for row in train_rows),
        "max_tokens": max(len(row["input_ids"]) for row in train_rows),
        "avg_tokens": sum(len(row["input_ids"]) for row in train_rows) / len(train_rows),
    }

    model.train()
    collator = make_causal_lm_data_collator(tokenizer)
    dataloader = DataLoader(
        train_rows,
        batch_size=config.per_device_train_batch_size,
        shuffle=config.shuffle_new_block_examples,
        generator=make_shuffle_generator(shuffle_seed) if config.shuffle_new_block_examples else None,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    log_history: list[dict[str, Any]] = []
    logging_loss_sum = 0.0
    logging_token_sum = 0
    total_weighted_loss = 0.0
    total_valid_tokens = 0
    global_step = global_step_start
    total_optimizer_steps = 0
    start_time = time.perf_counter()

    for epoch_idx in range(config.num_train_epochs_per_block):
        for batch in dataloader:
            global_step += 1
            total_optimizer_steps += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(**batch)
                loss = outputs.loss

            loss_value = float(loss.detach().cpu().item())
            valid_tokens = int((batch["labels"] != -100).sum().detach().cpu().item())

            if scaler is not None:
                scaler.scale(loss).backward()
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                optimizer.step()

            total_weighted_loss += loss_value * valid_tokens
            total_valid_tokens += valid_tokens
            logging_loss_sum += loss_value * valid_tokens
            logging_token_sum += valid_tokens

            if total_optimizer_steps % config.logging_steps == 0:
                window_loss = logging_loss_sum / max(logging_token_sum, 1)
                log_history.append(
                    {
                        "loss": window_loss,
                        "grad_norm": float(grad_norm.detach().cpu().item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": (epoch_idx + 1) / config.num_train_epochs_per_block,
                        "step": global_step,
                    }
                )
                logging_loss_sum = 0.0
                logging_token_sum = 0

    runtime_seconds = time.perf_counter() - start_time
    train_loss = total_weighted_loss / max(total_valid_tokens, 1)

    if logging_token_sum > 0:
        log_history.append(
            {
                "loss": logging_loss_sum / logging_token_sum,
                "grad_norm": None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": 1.0,
                "step": global_step,
            }
        )

    train_result_metrics = {
        "train_runtime": runtime_seconds,
        "train_samples_per_second": len(train_rows) / runtime_seconds if runtime_seconds > 0 else 0.0,
        "train_steps_per_second": total_optimizer_steps / runtime_seconds if runtime_seconds > 0 else 0.0,
        "train_loss": train_loss,
        "epoch": float(config.num_train_epochs_per_block),
        "optimizer_step_start": global_step_start,
        "optimizer_step_end": global_step,
        "optimizer_steps_this_block": total_optimizer_steps,
    }

    train_summary = {
        "run_name": run_name,
        "train_examples": len(train_rows),
        "tokenization_summary": tokenization_summary,
        "train_result_metrics": to_builtin(train_result_metrics),
        "trainer_log_history": to_builtin(log_history),
        "training_args": {
            "learning_rate": config.learning_rate,
            "num_train_epochs_per_block": config.num_train_epochs_per_block,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "logging_steps": config.logging_steps,
            "shuffle_new_block_examples": config.shuffle_new_block_examples,
            "optimizer": "AdamW",
            "lr_schedule": "constant",
        },
    }
    model.eval()
    return global_step, train_summary


def save_model_artifacts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    metadata: dict[str, Any],
    model_id: str,
) -> None:
    remove_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "base_model.txt").write_text(model_id, encoding="utf-8")
    write_json(output_dir / "model_summary.json", metadata)


def build_future_best_rows(rows: list[dict[str, Any]], loss_key: str) -> list[float]:
    future_best: list[float] = [0.0] * len(rows)
    best_so_far = float("inf")
    for idx in range(len(rows) - 1, -1, -1):
        best_so_far = min(best_so_far, float(rows[idx][loss_key]))
        future_best[idx] = best_so_far
    return future_best


def analyze_sweet_spot(
    rows: list[dict[str, Any]],
    loss_key: str,
    relative_headroom: float,
    absolute_loss_delta: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: row["train_game_count"])
    best_row = min(sorted_rows, key=lambda row: (float(row[loss_key]), row["train_game_count"]))
    future_best_losses = build_future_best_rows(sorted_rows, loss_key)
    best_loss = float(best_row[loss_key])
    headroom_threshold = max(absolute_loss_delta, best_loss * relative_headroom)

    recommended_row = sorted_rows[-1]
    headroom_rows = []
    for idx, row in enumerate(sorted_rows):
        future_best_loss = future_best_losses[idx]
        remaining_headroom = float(row[loss_key]) - future_best_loss
        headroom_rows.append(
            {
                "train_game_count": row["train_game_count"],
                "current_loss": float(row[loss_key]),
                "future_best_loss": future_best_loss,
                "remaining_headroom": remaining_headroom,
            }
        )
        if remaining_headroom <= headroom_threshold and recommended_row is sorted_rows[-1]:
            recommended_row = row

    analysis = {
        "loss_key": loss_key,
        "best_loss": best_loss,
        "headroom_threshold": headroom_threshold,
        "relative_headroom": relative_headroom,
        "absolute_loss_delta": absolute_loss_delta,
        "rows": headroom_rows,
    }
    return recommended_row, best_row, analysis


def select_candidate_counts(scan_rows: list[dict[str, Any]], config: ExperimentConfig) -> list[int]:
    if not scan_rows:
        raise ValueError("Cannot select candidates from an empty scan curve.")

    ordered_rows = sorted(scan_rows, key=lambda row: row["train_game_count"])
    ordered_counts = [int(row["train_game_count"]) for row in ordered_rows]
    count_to_index = {count: idx for idx, count in enumerate(ordered_counts)}

    best_scan_loss = min(float(row["scan_subset_loss"]) for row in ordered_rows)
    relative_margin_threshold = best_scan_loss * (1.0 + config.candidate_relative_margin)
    ranked_rows = sorted(ordered_rows, key=lambda row: (float(row["scan_subset_loss"]), row["train_game_count"]))

    scan_plateau_row, scan_best_row, _ = analyze_sweet_spot(
        ordered_rows,
        loss_key="scan_subset_loss",
        relative_headroom=config.sweet_spot_relative_headroom,
        absolute_loss_delta=config.sweet_spot_absolute_loss_delta,
    )

    candidate_counts = {
        int(scan_best_row["train_game_count"]),
        int(scan_plateau_row["train_game_count"]),
    }
    candidate_counts.update(int(row["train_game_count"]) for row in ranked_rows[: config.candidate_top_k])
    candidate_counts.update(
        int(row["train_game_count"])
        for row in ordered_rows
        if float(row["scan_subset_loss"]) <= relative_margin_threshold
    )

    expanded_counts = set(candidate_counts)
    for count in list(candidate_counts):
        center_idx = count_to_index[count]
        for offset in range(-config.candidate_neighbor_radius, config.candidate_neighbor_radius + 1):
            neighbor_idx = center_idx + offset
            if 0 <= neighbor_idx < len(ordered_counts):
                expanded_counts.add(ordered_counts[neighbor_idx])

    return sorted(expanded_counts)


def write_curve_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_train_game_cache(
    train_game_rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = []
    total_examples = 0
    total_tokens = 0
    max_tokens = 0
    min_tokens: int | None = None

    for game_idx, game in enumerate(train_game_rows, start=1):
        examples = build_examples_from_games([game], config.min_context_ply)
        tokenized_rows, tokenization_summary = prepare_tokenized_rows(examples, tokenizer, config.max_length)
        cache.append(
            {
                "game_index": game_idx,
                "game_id": game["id"],
                "example_count": len(tokenized_rows),
                "tokenized_rows": tokenized_rows,
                "tokenization_summary": tokenization_summary,
            }
        )
        total_examples += len(tokenized_rows)
        total_tokens += sum(len(row["input_ids"]) for row in tokenized_rows)
        if tokenized_rows:
            game_max = max(len(row["input_ids"]) for row in tokenized_rows)
            game_min = min(len(row["input_ids"]) for row in tokenized_rows)
            max_tokens = max(max_tokens, game_max)
            min_tokens = game_min if min_tokens is None else min(min_tokens, game_min)

    summary = {
        "train_games_cached": len(cache),
        "train_examples_total": total_examples,
        "train_avg_tokens": (total_tokens / total_examples) if total_examples else 0.0,
        "train_max_tokens": max_tokens,
        "train_min_tokens": min_tokens or 0,
    }
    return cache, summary


def select_block_rows_from_cache(
    train_game_cache: list[dict[str, Any]],
    previous_count: int,
    current_count: int,
) -> tuple[list[TokenizedRow], int]:
    if previous_count < 0 or current_count < 0:
        raise ValueError("Counts must be non-negative")
    if current_count < previous_count:
        raise ValueError("current_count must be at least previous_count")
    if current_count > len(train_game_cache):
        raise ValueError("Requested more training games than available in the ordered train pool")

    block = train_game_cache[previous_count:current_count]
    tokenized_rows = [row for game_item in block for row in game_item["tokenized_rows"]]
    return tokenized_rows, len(block)


def running_best(values: list[float]) -> list[float]:
    best_values: list[float] = []
    current_best = float("inf")
    for value in values:
        current_best = min(current_best, value)
        best_values.append(current_best)
    return best_values


def rolling_median(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)

    radius = max(0, window // 2)
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        smoothed.append(float(statistics.median(values[start:end])))
    return smoothed


def plot_scan_curve(
    results_dirs: dict[str, Path],
    scan_rows: list[dict[str, Any]],
    candidate_counts: list[int],
    recommended_count: int,
    best_count: int,
    best_loss: float,
    headroom_threshold: float,
    baseline_loss: float,
    smoothing_window: int,
) -> None:
    x_games = [row["train_game_count"] for row in scan_rows]
    y_scan_loss = [row["scan_subset_loss"] for row in scan_rows]
    y_running_best = running_best(y_scan_loss)
    y_rolling_median = rolling_median(y_scan_loss, smoothing_window)
    y_improvement = [baseline_loss - value for value in y_scan_loss]
    y_best_improvement = [baseline_loss - value for value in y_running_best]
    y_median_improvement = [baseline_loss - value for value in y_rolling_median]
    candidate_set = set(candidate_counts)
    candidate_x = [row["train_game_count"] for row in scan_rows if row["train_game_count"] in candidate_set]
    candidate_y = [row["scan_subset_loss"] for row in scan_rows if row["train_game_count"] in candidate_set]
    candidate_improvement_y = [baseline_loss - value for value in candidate_y]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_games,
        y_scan_loss,
        marker="o",
        linewidth=1.1,
        color="tab:gray",
        alpha=0.5,
        label="Raw scan loss",
    )
    plt.plot(
        x_games,
        y_rolling_median,
        linewidth=2.2,
        color="tab:blue",
        label=f"Rolling median ({smoothing_window})",
    )
    plt.plot(x_games, y_running_best, linewidth=2.0, color="tab:green", label="Best loss so far")
    plt.scatter(candidate_x, candidate_y, color="tab:orange", s=55, zorder=3, label="Full-validation candidates")
    plt.axhline(baseline_loss, color="tab:gray", linestyle="-.", linewidth=1.2, label="Baseline")
    plt.axhline(best_loss, color="tab:green", linestyle="--", linewidth=1.4, label="Best scan loss")
    plt.axhline(best_loss + headroom_threshold, color="tab:purple", linestyle=":", linewidth=1.4, label="Sweet-spot headroom")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best scan count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended count")
    plt.xlabel("Training games")
    plt.ylabel("Validation loss on scan subset")
    plt.title("Incremental scan loss vs number of training games")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "scan_subset_loss_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_games,
        y_improvement,
        marker="o",
        linewidth=1.1,
        color="tab:gray",
        alpha=0.5,
        label="Raw improvement vs baseline",
    )
    plt.plot(
        x_games,
        y_median_improvement,
        linewidth=2.2,
        color="tab:blue",
        label=f"Rolling median improvement ({smoothing_window})",
    )
    plt.plot(
        x_games,
        y_best_improvement,
        linewidth=2.0,
        color="tab:green",
        label="Best improvement so far",
    )
    plt.scatter(
        candidate_x,
        candidate_improvement_y,
        color="tab:orange",
        s=55,
        zorder=3,
        label="Full-validation candidates",
    )
    plt.axhline(0.0, color="tab:gray", linestyle="-.", linewidth=1.2, label="Baseline")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best scan count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended count")
    plt.xlabel("Training games")
    plt.ylabel("Validation improvement vs baseline")
    plt.title("Scan improvement vs number of training games")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "scan_subset_improvement_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def plot_full_validation_candidates(
    results_dirs: dict[str, Path],
    full_val_rows: list[dict[str, Any]],
    recommended_count: int,
    best_count: int,
    baseline_loss: float,
) -> None:
    if not full_val_rows:
        return

    rows = sorted(full_val_rows, key=lambda row: row["train_game_count"])
    x_games = [row["train_game_count"] for row in rows]
    y_loss = [row["full_val_loss"] for row in rows]
    y_running_best = running_best(y_loss)
    y_improvement = [baseline_loss - value for value in y_loss]
    y_best_improvement = [baseline_loss - value for value in y_running_best]

    plt.figure(figsize=(10, 6))
    plt.plot(x_games, y_loss, marker="o", linewidth=1.4, color="tab:blue", label="Full validation loss")
    plt.plot(x_games, y_running_best, linewidth=2.0, color="tab:green", label="Best full-validation loss so far")
    plt.axhline(baseline_loss, color="tab:gray", linestyle="-.", linewidth=1.2, label="Baseline")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best full-validation count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended count")
    plt.xlabel("Training games")
    plt.ylabel("Validation loss")
    plt.title("Full validation recheck on shortlisted checkpoints")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "full_validation_candidates_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x_games, y_improvement, marker="o", linewidth=1.4, color="tab:blue", label="Full validation improvement")
    plt.plot(
        x_games,
        y_best_improvement,
        linewidth=2.0,
        color="tab:green",
        label="Best full-validation improvement so far",
    )
    plt.axhline(0.0, color="tab:gray", linestyle="-.", linewidth=1.2, label="Baseline")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best full-validation count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended count")
    plt.xlabel("Training games")
    plt.ylabel("Validation improvement vs baseline")
    plt.title("Full validation improvement on shortlisted checkpoints")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "full_validation_improvement_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def plot_remaining_headroom(
    results_dirs: dict[str, Path],
    headroom_rows: list[dict[str, Any]],
    threshold: float,
    recommended_count: int,
) -> None:
    x_games = [row["train_game_count"] for row in headroom_rows]
    y_headroom = [row["remaining_headroom"] for row in headroom_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(x_games, y_headroom, marker="o", linewidth=1.8, label="Remaining possible improvement")
    plt.axhline(threshold, color="tab:orange", linestyle=":", linewidth=1.4, label="Sweet-spot threshold")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended count")
    plt.xlabel("Training games")
    plt.ylabel("Best later loss improvement still available")
    plt.title("Remaining validation headroom vs number of training games")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "remaining_headroom_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def build_player_result_row(result: dict[str, Any]) -> dict[str, Any]:
    final_summary = result["final_summary"]
    dataset_summary = final_summary["dataset_summary"]
    baseline = final_summary["baseline"]
    full_validation_selection = final_summary["full_validation_selection"]
    recommended_model = final_summary["recommended_model"]
    baseline_test_loss = float(baseline["test"]["loss"])
    recommended_test_loss = float(recommended_model["test"]["loss"])
    return {
        "username": dataset_summary["username"],
        "run_id": final_summary["run_id"],
        "results_dir": result["results_dir"],
        "train_games": dataset_summary["train_games"],
        "val_games": dataset_summary["val_games"],
        "test_games": dataset_summary["test_games"],
        "curve_train_games_used": dataset_summary["curve_train_games_used"],
        "baseline_scan_loss": float(baseline["val_scan_subset"]["loss"]),
        "baseline_val_loss": float(baseline["val_full"]["loss"]),
        "baseline_test_loss": baseline_test_loss,
        "best_val_count": int(full_validation_selection["best_validation_point"]["train_game_count"]),
        "best_val_loss": float(full_validation_selection["best_validation_point"]["val_loss"]),
        "recommended_count": int(full_validation_selection["sweet_spot"]["recommended_train_game_count"]),
        "recommended_val_loss": float(full_validation_selection["sweet_spot"]["recommended_val_loss"]),
        "recommended_test_loss": recommended_test_loss,
        "recommended_test_improvement_pct": (
            ((baseline_test_loss - recommended_test_loss) / baseline_test_loss) * 100.0 if baseline_test_loss > 0 else 0.0
        ),
    }


def aggregate_scan_curve_rows(player_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not player_results:
        raise ValueError("Cannot aggregate an empty list of player results.")

    row_maps = []
    common_counts: set[int] | None = None
    for player_result in player_results:
        username = player_result["final_summary"]["dataset_summary"]["username"]
        baseline_scan_loss = float(player_result["final_summary"]["baseline"]["val_scan_subset"]["loss"])
        rows_by_count = {int(row["train_game_count"]): row for row in player_result["scan_curve_rows"]}
        player_counts = set(rows_by_count.keys())
        common_counts = player_counts if common_counts is None else (common_counts & player_counts)
        row_maps.append(
            {
                "username": username,
                "baseline_scan_loss": baseline_scan_loss,
                "rows_by_count": rows_by_count,
            }
        )

    ordered_common_counts = sorted(common_counts or [])
    if not ordered_common_counts:
        return []

    aggregate_rows = []
    for count in ordered_common_counts:
        count_rows = [item["rows_by_count"][count] for item in row_maps]
        scan_losses = [float(row["scan_subset_loss"]) for row in count_rows]
        relative_improvements = [float(row["relative_scan_loss_reduction_pct"]) for row in count_rows]
        absolute_improvements = [
            float(item["baseline_scan_loss"]) - float(item["rows_by_count"][count]["scan_subset_loss"])
            for item in row_maps
        ]
        cumulative_examples = [float(row["cumulative_train_example_count"]) for row in count_rows]
        aggregate_rows.append(
            {
                "train_game_count": count,
                "player_count": len(count_rows),
                "avg_cumulative_train_example_count": float(statistics.mean(cumulative_examples)),
                "avg_scan_subset_loss": float(statistics.mean(scan_losses)),
                "std_scan_subset_loss": float(statistics.stdev(scan_losses)) if len(scan_losses) > 1 else 0.0,
                "min_scan_subset_loss": float(min(scan_losses)),
                "max_scan_subset_loss": float(max(scan_losses)),
                "avg_absolute_improvement_vs_baseline": float(statistics.mean(absolute_improvements)),
                "avg_relative_improvement_pct": float(statistics.mean(relative_improvements)),
            }
        )
    return aggregate_rows


def plot_aggregate_scan_curve(
    results_dirs: dict[str, Path],
    aggregate_rows: list[dict[str, Any]],
    recommended_count: int,
    best_count: int,
    baseline_loss: float,
    smoothing_window: int,
) -> None:
    if not aggregate_rows:
        return

    rows = sorted(aggregate_rows, key=lambda row: row["train_game_count"])
    x_games = [row["train_game_count"] for row in rows]
    y_loss = [row["avg_scan_subset_loss"] for row in rows]
    y_std = [row["std_scan_subset_loss"] for row in rows]
    y_running_best = running_best(y_loss)
    y_rolling_median = rolling_median(y_loss, smoothing_window)
    y_improvement = [baseline_loss - value for value in y_loss]
    y_best_improvement = [baseline_loss - value for value in y_running_best]
    y_median_improvement = [baseline_loss - value for value in y_rolling_median]
    lower_band = [max(0.0, loss - std) for loss, std in zip(y_loss, y_std)]
    upper_band = [loss + std for loss, std in zip(y_loss, y_std)]

    plt.figure(figsize=(10, 6))
    plt.plot(x_games, y_loss, marker="o", linewidth=1.4, color="tab:blue", label="Average scan loss")
    plt.fill_between(x_games, lower_band, upper_band, color="tab:blue", alpha=0.15, label="+/- 1 std")
    plt.plot(
        x_games,
        y_rolling_median,
        linewidth=2.0,
        color="tab:orange",
        label=f"Rolling median ({smoothing_window})",
    )
    plt.plot(x_games, y_running_best, linewidth=2.0, color="tab:green", label="Best average loss so far")
    plt.axhline(baseline_loss, color="tab:gray", linestyle="-.", linewidth=1.2, label="Average baseline")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best average count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended average count")
    plt.xlabel("Training games")
    plt.ylabel("Average validation loss")
    plt.title("Average scan loss across players")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "aggregate_scan_loss_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_games,
        y_improvement,
        marker="o",
        linewidth=1.4,
        color="tab:blue",
        label="Average improvement vs baseline",
    )
    plt.plot(
        x_games,
        y_median_improvement,
        linewidth=2.0,
        color="tab:orange",
        label=f"Rolling median improvement ({smoothing_window})",
    )
    plt.plot(
        x_games,
        y_best_improvement,
        linewidth=2.0,
        color="tab:green",
        label="Best average improvement so far",
    )
    plt.axhline(0.0, color="tab:gray", linestyle="-.", linewidth=1.2, label="Average baseline")
    plt.axvline(best_count, color="tab:green", linestyle="--", linewidth=1.2, label="Best average count")
    plt.axvline(recommended_count, color="tab:red", linestyle="--", linewidth=1.4, label="Recommended average count")
    plt.xlabel("Training games")
    plt.ylabel("Average validation improvement vs baseline")
    plt.title("Average scan improvement across players")
    plt.grid(alpha=0.3)
    plt.legend()
    plot_path = results_dirs["plots"] / "aggregate_scan_improvement_vs_games.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def _run_single_learning_curve_experiment(
    config: ExperimentConfig,
    run_id: str | None = None,
    root_override: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or make_run_id()
    results_dirs = ensure_results_dirs(config, run_id, root_override=root_override)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = hf_token if hf_token is not None else maybe_login_hf()
    seed_everything(config.split_seed)
    write_json(results_dirs["root"] / "config.json", asdict(config))

    print("Run ID:", run_id)
    print("Device:", device)
    print("Results root:", results_dirs["root"])
    print("Model:", config.model_id)

    raw_games = load_lichess_games_san(
        username=config.username,
        max_games=config.max_games,
        perf_type=config.perf_type,
        rated_only=config.rated_only,
    )
    parsed_games = parse_target_games(raw_games, config.username)
    train_games, val_games, test_games = split_games_train_val_test(
        parsed_games,
        split_seed=config.split_seed,
        split_strategy=config.split_strategy,
        test_frac=config.test_frac,
        val_frac_within_train=config.val_frac_within_train,
    )
    train_game_order = order_train_games_for_curve(train_games, config.split_strategy, config.split_seed)

    max_curve_games = len(train_game_order)
    if config.max_train_games_for_curve is not None:
        max_curve_games = min(max_curve_games, config.max_train_games_for_curve)
    min_curve_games = max(1, config.min_train_games)
    counts_to_scan = build_counts_to_scan(min_curve_games, max_curve_games, config.count_step_games)

    tokenizer = load_tokenizer(config.model_id, hf_token)

    val_full_examples = build_examples_from_games(val_games, config.min_context_ply)
    val_scan_examples, val_scan_subset_summary = select_eval_examples_subset(
        val_full_examples,
        max_examples=config.scan_val_subset_examples,
        seed=config.split_seed,
        subset_name="val_scan_subset",
    )
    test_examples = build_examples_from_games(test_games, config.min_context_ply)

    val_scan_rows, val_scan_tokenization_summary = prepare_tokenized_rows(val_scan_examples, tokenizer, config.max_length)
    val_full_rows, val_full_tokenization_summary = prepare_tokenized_rows(val_full_examples, tokenizer, config.max_length)
    test_rows, test_tokenization_summary = prepare_tokenized_rows(test_examples, tokenizer, config.max_length)

    train_game_cache, train_cache_summary = build_train_game_cache(
        train_game_order[:max_curve_games],
        tokenizer=tokenizer,
        config=config,
    )

    dataset_summary = {
        "run_id": run_id,
        "username": config.username,
        "perf_type": config.perf_type,
        "raw_games_loaded": len(raw_games),
        "parsed_games_used": len(parsed_games),
        "split_strategy": config.split_strategy,
        "train_games": len(train_games),
        "val_games": len(val_games),
        "test_games": len(test_games),
        "curve_train_games_used": max_curve_games,
        "val_examples_full": len(val_full_examples),
        "val_examples_scan_subset": len(val_scan_examples),
        "test_examples": len(test_examples),
        "counts_to_scan": counts_to_scan,
        "count_step_games": config.count_step_games,
        "min_context_ply": config.min_context_ply,
        "curve_metric": "validation_loss",
        "curve_mode": "incremental_continue_training_constant_lr",
        "sweet_spot_relative_headroom": config.sweet_spot_relative_headroom,
        "sweet_spot_absolute_loss_delta": config.sweet_spot_absolute_loss_delta,
        "candidate_top_k": config.candidate_top_k,
        "candidate_relative_margin": config.candidate_relative_margin,
        "val_scan_subset_summary": val_scan_subset_summary,
        "train_cache_summary": train_cache_summary,
    }
    write_json(results_dirs["root"] / "dataset_summary.json", dataset_summary)

    print("Loaded raw games:", len(raw_games))
    print("Parsed games:", len(parsed_games))
    print("Train / val / test games:", len(train_games), len(val_games), len(test_games))
    print("Curve train games used:", max_curve_games)
    print("Val examples (full / scan subset):", len(val_full_examples), len(val_scan_examples))
    print("Test examples:", len(test_examples))
    print("Counts to scan:", counts_to_scan)

    baseline_model = load_base_model(config.model_id, tokenizer, device, hf_token)
    baseline_scan_metrics = evaluate_lm_loss(
        model=baseline_model,
        tokenizer=tokenizer,
        eval_rows=val_scan_rows,
        batch_size=config.per_device_eval_batch_size,
        device=device,
        run_name="baseline_val_scan_subset",
    )
    baseline_val_metrics = evaluate_lm_loss(
        model=baseline_model,
        tokenizer=tokenizer,
        eval_rows=val_full_rows,
        batch_size=config.per_device_eval_batch_size,
        device=device,
        run_name="baseline_val_full",
    )
    baseline_test_metrics = evaluate_lm_loss(
        model=baseline_model,
        tokenizer=tokenizer,
        eval_rows=test_rows,
        batch_size=config.per_device_eval_batch_size,
        device=device,
        run_name="baseline_test",
    )
    write_json(results_dirs["root"] / "baseline_scan_metrics.json", baseline_scan_metrics)
    write_json(results_dirs["root"] / "baseline_val_metrics.json", baseline_val_metrics)
    write_json(results_dirs["root"] / "baseline_test_metrics.json", baseline_test_metrics)
    print("Baseline scan subset loss:", round(baseline_scan_metrics["loss"], 4))
    print("Baseline full validation loss:", round(baseline_val_metrics["loss"], 4))
    print("Baseline test loss:", round(baseline_test_metrics["loss"], 4))
    cleanup_torch_objects(baseline_model)
    del baseline_model

    current_model = build_lora_model(
        model_id=config.model_id,
        tokenizer=tokenizer,
        device=device,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        token=hf_token,
    )
    optimizer = AdamW(
        [parameter for parameter in current_model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    scan_curve_rows = []
    trainer_logs = []
    previous_count = 0
    cumulative_train_examples = 0
    global_step = 0

    for train_game_count in counts_to_scan:
        new_block_rows, new_block_game_count = select_block_rows_from_cache(
            train_game_cache,
            previous_count=previous_count,
            current_count=train_game_count,
        )
        run_name = f"games_{train_game_count:04d}"
        print(
            f"continuing {run_name}: adding {new_block_game_count} games / {len(new_block_rows)} examples "
            f"(cumulative games={train_game_count})"
        )

        shuffle_seed = config.split_seed + train_game_count
        global_step, train_summary = train_incremental_block(
            model=current_model,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            train_rows=new_block_rows,
            config=config,
            device=device,
            run_name=run_name,
            global_step_start=global_step,
            shuffle_seed=shuffle_seed,
        )
        trainer_logs.append(train_summary)
        cumulative_train_examples += len(new_block_rows)

        scan_subset_metrics = evaluate_lm_loss(
            model=current_model,
            tokenizer=tokenizer,
            eval_rows=val_scan_rows,
            batch_size=config.per_device_eval_batch_size,
            device=device,
            run_name=run_name,
        )

        checkpoint_dir = results_dirs["curve_checkpoints"] / run_name
        save_model_artifacts(
            model=current_model,
            tokenizer=tokenizer,
            output_dir=checkpoint_dir,
            metadata={
                "train_game_count": train_game_count,
                "new_block_game_count": new_block_game_count,
                "new_block_example_count": len(new_block_rows),
                "cumulative_train_examples": cumulative_train_examples,
                "scan_subset_metrics": scan_subset_metrics,
                "train_summary": train_summary,
            },
            model_id=config.model_id,
        )

        curve_row = {
            "train_game_count": train_game_count,
            "new_block_game_count": new_block_game_count,
            "new_block_example_count": len(new_block_rows),
            "cumulative_train_example_count": cumulative_train_examples,
            "scan_subset_loss": scan_subset_metrics["loss"],
            "scan_subset_perplexity": scan_subset_metrics["perplexity"],
            "delta_scan_loss_vs_baseline": scan_subset_metrics["loss"] - baseline_scan_metrics["loss"],
            "relative_scan_loss_reduction_pct": (
                (baseline_scan_metrics["loss"] - scan_subset_metrics["loss"]) / baseline_scan_metrics["loss"]
            )
            * 100.0,
            "train_runtime_seconds": train_summary["train_result_metrics"].get("train_runtime"),
            "final_train_loss": train_summary["train_result_metrics"].get("train_loss"),
            "optimizer_step_end": train_summary["train_result_metrics"].get("optimizer_step_end"),
            "checkpoint_dir": str(checkpoint_dir),
            "train_summary": train_summary,
            "raw_scan_subset_metrics": scan_subset_metrics,
        }
        scan_curve_rows.append(curve_row)

        write_json(results_dirs["root"] / "curve_results.json", scan_curve_rows)
        write_json(results_dirs["root"] / "trainer_logs.json", trainer_logs)

        print(
            "result:",
            {
                "games": train_game_count,
                "scan_subset_loss": round(curve_row["scan_subset_loss"], 4),
                "scan_subset_loss_reduction_pct": round(curve_row["relative_scan_loss_reduction_pct"], 2),
            },
        )

        previous_count = train_game_count

    cleanup_torch_objects(current_model)
    del current_model, optimizer, scaler

    if not scan_curve_rows:
        raise ValueError("No curve results were generated. Check the dataset size and count configuration.")

    scan_curve_rows = sorted(scan_curve_rows, key=lambda row: row["train_game_count"])
    candidate_counts = select_candidate_counts(scan_curve_rows, config)
    rows_by_count = {row["train_game_count"]: row for row in scan_curve_rows}

    scan_recommended_row, scan_best_row, scan_headroom_analysis = analyze_sweet_spot(
        scan_curve_rows,
        loss_key="scan_subset_loss",
        relative_headroom=config.sweet_spot_relative_headroom,
        absolute_loss_delta=config.sweet_spot_absolute_loss_delta,
    )

    print("Shortlisted counts for full validation:", candidate_counts)
    full_validation_rows = []
    for candidate_count in candidate_counts:
        candidate_scan_row = rows_by_count[candidate_count]
        candidate_model = load_lora_checkpoint_model(
            model_id=config.model_id,
            tokenizer=tokenizer,
            checkpoint_dir=Path(candidate_scan_row["checkpoint_dir"]),
            device=device,
            token=hf_token,
        )
        full_val_metrics = evaluate_lm_loss(
            model=candidate_model,
            tokenizer=tokenizer,
            eval_rows=val_full_rows,
            batch_size=config.per_device_eval_batch_size,
            device=device,
            run_name=f"full_val_games_{candidate_count:04d}",
        )
        full_validation_row = {
            "train_game_count": candidate_count,
            "cumulative_train_example_count": candidate_scan_row["cumulative_train_example_count"],
            "scan_subset_loss": candidate_scan_row["scan_subset_loss"],
            "full_val_loss": full_val_metrics["loss"],
            "full_val_perplexity": full_val_metrics["perplexity"],
            "checkpoint_dir": candidate_scan_row["checkpoint_dir"],
            "raw_full_val_metrics": full_val_metrics,
        }
        full_validation_rows.append(full_validation_row)
        cleanup_torch_objects(candidate_model)
        del candidate_model

    full_validation_rows = sorted(full_validation_rows, key=lambda row: row["train_game_count"])
    write_json(results_dirs["root"] / "full_validation_candidates.json", full_validation_rows)

    recommended_full_val_row, best_full_val_row, full_val_headroom_analysis = analyze_sweet_spot(
        full_validation_rows,
        loss_key="full_val_loss",
        relative_headroom=config.sweet_spot_relative_headroom,
        absolute_loss_delta=config.sweet_spot_absolute_loss_delta,
    )

    recommended_model = load_lora_checkpoint_model(
        model_id=config.model_id,
        tokenizer=tokenizer,
        checkpoint_dir=Path(recommended_full_val_row["checkpoint_dir"]),
        device=device,
        token=hf_token,
    )
    recommended_test_metrics = evaluate_lm_loss(
        model=recommended_model,
        tokenizer=tokenizer,
        eval_rows=test_rows,
        batch_size=config.per_device_eval_batch_size,
        device=device,
        run_name="recommended_test",
    )

    final_summary = {
        "mode": "single_player",
        "run_id": run_id,
        "config": asdict(config),
        "dataset_summary": dataset_summary,
        "tokenization": {
            "val_scan_subset": val_scan_tokenization_summary,
            "val_full": val_full_tokenization_summary,
            "test": test_tokenization_summary,
        },
        "baseline": {
            "val_scan_subset": baseline_scan_metrics,
            "val_full": baseline_val_metrics,
            "test": baseline_test_metrics,
        },
        "scan_curve": {
            "best_validation_point": {
                "train_game_count": scan_best_row["train_game_count"],
                "cumulative_train_example_count": scan_best_row["cumulative_train_example_count"],
                "scan_subset_loss": scan_best_row["scan_subset_loss"],
                "scan_subset_perplexity": scan_best_row["scan_subset_perplexity"],
                "checkpoint_dir": scan_best_row["checkpoint_dir"],
            },
            "scan_recommended_point": {
                "train_game_count": scan_recommended_row["train_game_count"],
                "scan_subset_loss": scan_recommended_row["scan_subset_loss"],
                "checkpoint_dir": scan_recommended_row["checkpoint_dir"],
            },
            "headroom_analysis": scan_headroom_analysis,
            "candidate_counts_for_full_validation": candidate_counts,
        },
        "full_validation_selection": {
            "best_validation_point": {
                "train_game_count": best_full_val_row["train_game_count"],
                "cumulative_train_example_count": best_full_val_row["cumulative_train_example_count"],
                "val_loss": best_full_val_row["full_val_loss"],
                "val_perplexity": best_full_val_row["full_val_perplexity"],
                "checkpoint_dir": best_full_val_row["checkpoint_dir"],
            },
            "sweet_spot": {
                "metric": "validation_loss",
                "curve_mode": "incremental_continue_training_constant_lr",
                "selection": "earliest_count_with_low_remaining_headroom_after_full_validation_recheck",
                "sweet_spot_relative_headroom": config.sweet_spot_relative_headroom,
                "sweet_spot_absolute_loss_delta": config.sweet_spot_absolute_loss_delta,
                "headroom_threshold": full_val_headroom_analysis["headroom_threshold"],
                "recommended_train_game_count": recommended_full_val_row["train_game_count"],
                "recommended_train_example_count": recommended_full_val_row["cumulative_train_example_count"],
                "recommended_val_loss": recommended_full_val_row["full_val_loss"],
                "recommended_val_perplexity": recommended_full_val_row["full_val_perplexity"],
                "recommended_checkpoint_dir": recommended_full_val_row["checkpoint_dir"],
                "interpretation": (
                    f"The sweet spot is {recommended_full_val_row['train_game_count']} games because, from this point on, "
                    f"the best later checkpoint improves validation loss by at most "
                    f"{full_val_headroom_analysis['headroom_threshold']:.4f}."
                ),
            },
            "headroom_analysis": full_val_headroom_analysis,
            "candidate_rows": full_validation_rows,
        },
        "recommended_model": {
            "train_game_count": recommended_full_val_row["train_game_count"],
            "test": recommended_test_metrics,
        },
    }
    write_json(results_dirs["root"] / "final_summary.json", final_summary)

    write_curve_csv(
        results_dirs["root"] / "curve_results.csv",
        scan_curve_rows,
        fieldnames=[
            "train_game_count",
            "new_block_game_count",
            "new_block_example_count",
            "cumulative_train_example_count",
            "scan_subset_loss",
            "scan_subset_perplexity",
            "delta_scan_loss_vs_baseline",
            "relative_scan_loss_reduction_pct",
            "train_runtime_seconds",
            "final_train_loss",
            "optimizer_step_end",
            "checkpoint_dir",
        ],
    )
    write_curve_csv(
        results_dirs["root"] / "full_validation_candidates.csv",
        full_validation_rows,
        fieldnames=[
            "train_game_count",
            "cumulative_train_example_count",
            "scan_subset_loss",
            "full_val_loss",
            "full_val_perplexity",
            "checkpoint_dir",
        ],
    )

    save_model_artifacts(
        model=recommended_model,
        tokenizer=tokenizer,
        output_dir=results_dirs["best_model"],
        metadata=final_summary,
        model_id=config.model_id,
    )

    plot_scan_curve(
        results_dirs=results_dirs,
        scan_rows=scan_curve_rows,
        candidate_counts=candidate_counts,
        recommended_count=recommended_full_val_row["train_game_count"],
        best_count=scan_best_row["train_game_count"],
        best_loss=scan_best_row["scan_subset_loss"],
        headroom_threshold=full_val_headroom_analysis["headroom_threshold"],
        baseline_loss=baseline_scan_metrics["loss"],
        smoothing_window=config.plot_smoothing_window,
    )
    plot_full_validation_candidates(
        results_dirs=results_dirs,
        full_val_rows=full_validation_rows,
        recommended_count=recommended_full_val_row["train_game_count"],
        best_count=best_full_val_row["train_game_count"],
        baseline_loss=baseline_val_metrics["loss"],
    )
    plot_remaining_headroom(
        results_dirs=results_dirs,
        headroom_rows=full_val_headroom_analysis["rows"],
        threshold=full_val_headroom_analysis["headroom_threshold"],
        recommended_count=recommended_full_val_row["train_game_count"],
    )

    print("Best full-validation count:", best_full_val_row["train_game_count"])
    print("Recommended sweet-spot count:", recommended_full_val_row["train_game_count"])
    print("Recommended test loss:", round(recommended_test_metrics["loss"], 4))
    print("Saved final summary to:", results_dirs["root"] / "final_summary.json")

    cleanup_torch_objects(recommended_model)
    del recommended_model

    return {
        "run_id": run_id,
        "results_dir": str(results_dirs["root"]),
        "scan_curve_rows": scan_curve_rows,
        "full_validation_rows": full_validation_rows,
        "final_summary": final_summary,
    }


def run_multi_player_learning_curve_experiment(config: ExperimentConfig) -> dict[str, Any]:
    usernames = resolve_usernames(config)
    if len(usernames) < 2:
        single_config = replace(config, username=usernames[0], usernames=None)
        return _run_single_learning_curve_experiment(single_config)

    run_id = make_run_id()
    results_dirs = ensure_multi_results_dirs(config, run_id, usernames)
    hf_token = maybe_login_hf()
    seed_everything(config.split_seed)
    write_json(results_dirs["root"] / "config.json", asdict(config))

    print("Run ID:", run_id)
    print("Mode: multi-player average")
    print("Players:", usernames)
    print("Results root:", results_dirs["root"])

    player_results: list[dict[str, Any]] = []
    player_summary_rows: list[dict[str, Any]] = []
    failed_players: list[dict[str, str]] = []

    for username in usernames:
        print(f"\n=== Running player: {username} ===")
        player_config = replace(config, username=username, usernames=None)
        player_root = results_dirs["players"] / f"{username}_{config.perf_type}"
        try:
            player_result = _run_single_learning_curve_experiment(
                player_config,
                run_id=run_id,
                root_override=player_root,
                hf_token=hf_token,
            )
        except Exception as exc:
            failed_players.append({"username": username, "error": str(exc)})
            print(f"Skipping {username}: {exc}")
            continue

        player_results.append(player_result)
        player_summary_rows.append(build_player_result_row(player_result))

    if not player_results:
        raise ValueError("No player runs completed successfully.")

    aggregate_player_results = list(player_results)
    aggregate_player_summary_rows = list(player_summary_rows)
    excluded_from_aggregate_players: list[dict[str, Any]] = []
    required_curve_games = config.multi_player_min_curve_games_for_aggregate

    if required_curve_games is not None:
        aggregate_player_results = []
        aggregate_player_summary_rows = []
        for player_result, player_summary in zip(player_results, player_summary_rows):
            curve_train_games_used = int(player_summary["curve_train_games_used"])
            if curve_train_games_used >= required_curve_games:
                aggregate_player_results.append(player_result)
                aggregate_player_summary_rows.append(player_summary)
                continue
            excluded_from_aggregate_players.append(
                {
                    "username": player_summary["username"],
                    "curve_train_games_used": curve_train_games_used,
                    "required_curve_games": required_curve_games,
                    "reason": "insufficient_curve_train_games_for_aggregate",
                }
            )

        if len(aggregate_player_results) < 2:
            available_counts = ", ".join(
                f"{row['username']}={row['curve_train_games_used']}" for row in player_summary_rows
            )
            raise ValueError(
                "Fewer than two players meet multi_player_min_curve_games_for_aggregate="
                f"{required_curve_games}. Available curve_train_games_used: {available_counts}"
            )

        kept_usernames = [row["username"] for row in aggregate_player_summary_rows]
        print(
            "Aggregate filter:",
            {
                "required_curve_games": required_curve_games,
                "kept_players": kept_usernames,
                "excluded_players": excluded_from_aggregate_players,
            },
        )

    aggregate_scan_rows = aggregate_scan_curve_rows(aggregate_player_results)
    if not aggregate_scan_rows:
        raise ValueError("The player runs do not share any common train-game counts to aggregate.")

    aggregate_recommended_row, aggregate_best_row, aggregate_headroom_analysis = analyze_sweet_spot(
        aggregate_scan_rows,
        loss_key="avg_scan_subset_loss",
        relative_headroom=config.sweet_spot_relative_headroom,
        absolute_loss_delta=config.sweet_spot_absolute_loss_delta,
    )

    baseline_scan_stats = summarize_numeric([row["baseline_scan_loss"] for row in aggregate_player_summary_rows])
    baseline_val_stats = summarize_numeric([row["baseline_val_loss"] for row in aggregate_player_summary_rows])
    baseline_test_stats = summarize_numeric([row["baseline_test_loss"] for row in aggregate_player_summary_rows])
    recommended_count_stats = summarize_numeric([float(row["recommended_count"]) for row in aggregate_player_summary_rows])
    recommended_val_stats = summarize_numeric([row["recommended_val_loss"] for row in aggregate_player_summary_rows])
    recommended_test_stats = summarize_numeric([row["recommended_test_loss"] for row in aggregate_player_summary_rows])
    recommended_test_improvement_stats = summarize_numeric(
        [row["recommended_test_improvement_pct"] for row in aggregate_player_summary_rows]
    )
    best_val_count_stats = summarize_numeric([float(row["best_val_count"]) for row in aggregate_player_summary_rows])
    best_val_loss_stats = summarize_numeric([row["best_val_loss"] for row in aggregate_player_summary_rows])

    aggregate_summary = {
        "mode": "multi_player_average",
        "run_id": run_id,
        "config": asdict(config),
        "requested_player_count": len(usernames),
        "successful_player_count": len(player_results),
        "aggregate_player_count": len(aggregate_player_results),
        "failed_player_count": len(failed_players),
        "successful_usernames": [row["username"] for row in player_summary_rows],
        "aggregate_usernames": [row["username"] for row in aggregate_player_summary_rows],
        "failed_players": failed_players,
        "excluded_from_aggregate_players": excluded_from_aggregate_players,
        "player_summaries": player_summary_rows,
        "aggregate_scan_curve": {
            "curve_metric": "average_scan_subset_loss_on_common_counts",
            "required_curve_games_for_aggregate": required_curve_games,
            "common_counts_to_scan": [row["train_game_count"] for row in aggregate_scan_rows],
            "player_count_per_point": aggregate_scan_rows[0]["player_count"],
            "average_baseline_scan_loss": baseline_scan_stats["mean"],
            "best_average_point": {
                "train_game_count": aggregate_best_row["train_game_count"],
                "avg_scan_subset_loss": aggregate_best_row["avg_scan_subset_loss"],
                "avg_relative_improvement_pct": aggregate_best_row["avg_relative_improvement_pct"],
            },
            "sweet_spot": {
                "recommended_train_game_count": aggregate_recommended_row["train_game_count"],
                "recommended_avg_scan_subset_loss": aggregate_recommended_row["avg_scan_subset_loss"],
                "recommended_avg_relative_improvement_pct": aggregate_recommended_row["avg_relative_improvement_pct"],
                "headroom_threshold": aggregate_headroom_analysis["headroom_threshold"],
                "interpretation": (
                    f"The aggregate sweet spot is {aggregate_recommended_row['train_game_count']} games because, "
                    f"from this point on, the best later average scan loss improves by at most "
                    f"{aggregate_headroom_analysis['headroom_threshold']:.4f}."
                ),
            },
            "headroom_analysis": aggregate_headroom_analysis,
            "rows": aggregate_scan_rows,
        },
        "aggregate_recommended_models": {
            "average_recommended_count": recommended_count_stats["mean"],
            "median_recommended_count": recommended_count_stats["median"],
            "average_best_validation_count": best_val_count_stats["mean"],
            "median_best_validation_count": best_val_count_stats["median"],
            "average_recommended_val_loss": recommended_val_stats["mean"],
            "average_best_val_loss": best_val_loss_stats["mean"],
            "average_recommended_test_loss": recommended_test_stats["mean"],
            "average_relative_test_improvement_pct": recommended_test_improvement_stats["mean"],
            "baseline_scan_loss_stats": baseline_scan_stats,
            "baseline_val_loss_stats": baseline_val_stats,
            "baseline_test_loss_stats": baseline_test_stats,
            "recommended_val_loss_stats": recommended_val_stats,
            "recommended_test_loss_stats": recommended_test_stats,
            "recommended_test_improvement_pct_stats": recommended_test_improvement_stats,
        },
    }

    write_json(results_dirs["root"] / "player_summaries.json", player_summary_rows)
    write_json(results_dirs["root"] / "aggregate_scan_curve.json", aggregate_scan_rows)
    write_json(results_dirs["root"] / "final_summary.json", aggregate_summary)

    write_curve_csv(
        results_dirs["root"] / "player_summary.csv",
        player_summary_rows,
        fieldnames=[
            "username",
            "run_id",
            "results_dir",
            "train_games",
            "val_games",
            "test_games",
            "curve_train_games_used",
            "baseline_scan_loss",
            "baseline_val_loss",
            "baseline_test_loss",
            "best_val_count",
            "best_val_loss",
            "recommended_count",
            "recommended_val_loss",
            "recommended_test_loss",
            "recommended_test_improvement_pct",
        ],
    )
    write_curve_csv(
        results_dirs["root"] / "aggregate_scan_curve.csv",
        aggregate_scan_rows,
        fieldnames=[
            "train_game_count",
            "player_count",
            "avg_cumulative_train_example_count",
            "avg_scan_subset_loss",
            "std_scan_subset_loss",
            "min_scan_subset_loss",
            "max_scan_subset_loss",
            "avg_absolute_improvement_vs_baseline",
            "avg_relative_improvement_pct",
        ],
    )

    plot_aggregate_scan_curve(
        results_dirs=results_dirs,
        aggregate_rows=aggregate_scan_rows,
        recommended_count=aggregate_recommended_row["train_game_count"],
        best_count=aggregate_best_row["train_game_count"],
        baseline_loss=baseline_scan_stats["mean"],
        smoothing_window=config.plot_smoothing_window,
    )
    plot_remaining_headroom(
        results_dirs=results_dirs,
        headroom_rows=aggregate_headroom_analysis["rows"],
        threshold=aggregate_headroom_analysis["headroom_threshold"],
        recommended_count=aggregate_recommended_row["train_game_count"],
    )

    print("\nAggregate average sweet spot:", aggregate_recommended_row["train_game_count"])
    print("Average recommended test improvement (%):", round(recommended_test_improvement_stats["mean"], 2))
    print("Saved aggregate summary to:", results_dirs["root"] / "final_summary.json")

    return {
        "run_id": run_id,
        "results_dir": str(results_dirs["root"]),
        "player_results": player_results,
        "player_summaries": player_summary_rows,
        "aggregate_scan_rows": aggregate_scan_rows,
        "scan_curve_rows": aggregate_scan_rows,
        "full_validation_rows": [],
        "final_summary": aggregate_summary,
    }


def run_learning_curve_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    usernames = resolve_usernames(config)
    if len(usernames) == 1:
        single_config = replace(config, username=usernames[0], usernames=None)
        return _run_single_learning_curve_experiment(single_config)
    return run_multi_player_learning_curve_experiment(config)
