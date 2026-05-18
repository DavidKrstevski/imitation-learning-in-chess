from __future__ import annotations

import gc
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import requests
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from backend.player_policy import (
    DEFAULT_MAX_CONTEXT_MOVES,
    assert_no_overlap,
    build_position_examples,
    build_prompt,
    build_scoring_batch,
    denormalize_move_uci,
    evaluate_ranker,
    normalize_board,
    pick_top_wrong_moves,
    score_legal_moves,
    select_reranker_negatives,
    split_games_game_level,
)


@dataclass
class Model3ExperimentConfig:
    username: str = "Lance5500"
    perf_type: str = "classical"
    max_games: int = 2000
    rated_only: bool = True
    random_seed: int = 42
    test_frac: float = 0.2
    val_frac_within_train: float = 0.2
    min_context_candidates: tuple[int, ...] = (0, 8)
    max_context_moves: int = DEFAULT_MAX_CONTEXT_MOVES
    top_k_eval: int = 5
    model_id: str = "daavidhauser/chess-bot-3000-250m"
    metric_goal_relative_top1: float = 1.10
    stage1_first_sweep_epochs: tuple[int, ...] = (1, 3)
    stage1_first_sweep_lrs: tuple[float, ...] = (5e-5, 1e-4, 2e-4)
    stage1_refine_ranks: tuple[int, ...] = (8, 16, 32)
    stage1_lora_alpha: int = 32
    stage1_lora_dropout: float = 0.05
    stage1_target_modules: str = "all-linear"
    stage1_max_length: int = 320
    stage1_batch_size: int | None = None
    reranker_uniform_negatives: int = 4
    reranker_hard_negatives: int = 3
    reranker_gradient_accumulation_steps: int = 4

    def resolved_stage1_batch_size(self) -> int:
        if self.stage1_batch_size is not None:
            return int(self.stage1_batch_size)
        return 4 if torch.cuda.is_available() else 1


class Model3Experiment:
    def __init__(self, config: Model3ExperimentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None
        self.run_id = self.make_run_id()
        self.output_dirs = self.make_output_dirs(self.run_id)

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def maybe_login_hf(self) -> str | None:
        token = os.getenv("HF_TOKEN")
        if not token:
            return None
        try:
            login(token=token, add_to_git_credential=False)
            print("Hugging Face token detected and login attempted.")
        except Exception as exc:
            print("HF login warning:", exc)
        return token

    def make_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def make_output_dirs(self, run_id: str) -> dict[str, Path]:
        root = Path("outputs") / "model3_runs" / run_id
        dirs = {
            "root": root,
            "scratch": root / "scratch",
            "models": root / "models",
            "logs": Path("outputs") / "experiment_logs",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def to_builtin(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self.to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.to_builtin(v) for v in value]
        if isinstance(value, Counter):
            return dict(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def write_experiment_log(self, payload: dict[str, Any]) -> str:
        log_path = self.output_dirs["logs"] / (
            f"{self.config.username}_{self.config.perf_type}_model3_{self.run_id}.json"
        )
        log_path.write_text(json.dumps(self.to_builtin(payload), indent=2), encoding="utf-8")
        return str(log_path)

    def cleanup_memory(self, *objects: Any) -> None:
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

    def save_model_bundle(self, model_bundle: dict[str, Any], tokenizer, save_dir: Path) -> str:
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir / "tokenizer")
        for name, model in model_bundle.items():
            model.save_pretrained(save_dir / name)
        return str(save_dir)

    def sort_metric_rows(self, rows: list[dict[str, Any]], metric_prefix: str = "val") -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                -row[f"{metric_prefix}_top1"],
                -row[f"{metric_prefix}_top5"],
                row.get("adapter_mode", ""),
                row.get("lora_rank", 0),
                row.get("learning_rate", 0.0),
                row.get("num_train_epochs", 0),
            ),
        )

    def print_header(self) -> None:
        print("Run ID:", self.run_id)
        print("Device:", self.device)
        print(
            "Player:",
            self.config.username,
            "| Perf:",
            self.config.perf_type,
            "| Rated only:",
            self.config.rated_only,
        )

    def load_lichess_games_san(self) -> list[dict[str, Any]]:
        url = f"https://lichess.org/api/games/user/{self.config.username}"
        headers = {"Accept": "application/x-ndjson"}
        params = {
            "max": self.config.max_games,
            "moves": "true",
            "pgnInJson": "false",
            "opening": "true",
            "clocks": "false",
            "evals": "false",
            "perfType": self.config.perf_type,
        }
        if self.config.rated_only:
            params["rated"] = "true"

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        raw_games = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        rows: list[dict[str, Any]] = []
        for idx, game in enumerate(raw_games):
            game_perf = game.get("perf") or game.get("speed")
            if game_perf != self.config.perf_type:
                continue
            rows.append(
                {
                    "id": game.get("id") or f"game_{idx}",
                    "perf": game_perf,
                    "rated": game.get("rated"),
                    "white": game.get("players", {}).get("white", {}).get("user", {}).get("name"),
                    "black": game.get("players", {}).get("black", {}).get("user", {}).get("name"),
                    "opening_name": (game.get("opening") or {}).get("name"),
                    "moves_san": game.get("moves", "").split(),
                }
            )
        return rows

    def parse_games_for_player(self, raw_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        username_lower = self.config.username.lower()
        parsed: list[dict[str, Any]] = []
        for row in raw_games:
            white = (row.get("white") or "").lower()
            black = (row.get("black") or "").lower()
            if white == username_lower:
                user_color = "white"
            elif black == username_lower:
                user_color = "black"
            else:
                continue

            board = chess.Board()
            uci_moves: list[str] = []
            valid_game = True
            for san in row.get("moves_san", []):
                try:
                    move = board.parse_san(san)
                except Exception:
                    valid_game = False
                    break
                uci_moves.append(move.uci())
                board.push(move)

            if valid_game and len(uci_moves) >= 2:
                parsed.append(
                    {
                        "id": row["id"],
                        "user_color": user_color,
                        "opening_name": row.get("opening_name"),
                        "uci_moves": uci_moves,
                    }
                )
        return parsed

    def build_bundle_for_min_context(
        self,
        train_core_games: list[dict[str, Any]],
        val_games: list[dict[str, Any]],
        test_games: list[dict[str, Any]],
        min_context_ply: int,
    ) -> dict[str, Any]:
        train_full_games = list(train_core_games) + list(val_games)
        return {
            "min_context_ply": min_context_ply,
            "train_core_games": list(train_core_games),
            "val_games": list(val_games),
            "test_games": list(test_games),
            "train_full_games": train_full_games,
            "train_core_examples": build_position_examples(
                train_core_games,
                min_context_ply=min_context_ply,
                max_context_moves=self.config.max_context_moves,
            ),
            "val_examples": build_position_examples(
                val_games,
                min_context_ply=min_context_ply,
                max_context_moves=self.config.max_context_moves,
            ),
            "train_full_examples": build_position_examples(
                train_full_games,
                min_context_ply=min_context_ply,
                max_context_moves=self.config.max_context_moves,
            ),
            "test_examples": build_position_examples(
                test_games,
                min_context_ply=min_context_ply,
                max_context_moves=self.config.max_context_moves,
            ),
        }

    def run_correctness_checks(self, bundle: dict[str, Any]) -> dict[str, bool]:
        assert_no_overlap(
            {
                "train": bundle["train_core_games"],
                "val": bundle["val_games"],
                "test": bundle["test_games"],
            }
        )

        checked_examples = (
            bundle["train_core_examples"][:25]
            + bundle["val_examples"][:25]
            + bundle["test_examples"][:25]
        )
        target_legal_ok = True
        prompt_ok = True
        black_roundtrip_ok = True
        for ex in checked_examples:
            if ex["norm_target"] not in ex["legal_moves_norm"]:
                target_legal_ok = False
            if not ex["prompt"].startswith("FEN: "):
                prompt_ok = False
            if ex["side"] == "black":
                roundtrip = denormalize_move_uci(ex["norm_target"], ex["side"])
                if roundtrip != ex["orig_target"]:
                    black_roundtrip_ok = False

        board = chess.Board()
        mirrored = normalize_board(board, "black")
        board_roundtrip_ok = normalize_board(mirrored, "black").board_fen() == board.board_fen()

        return {
            "target_is_legal": target_legal_ok,
            "prompt_format_ok": prompt_ok,
            "black_move_roundtrip_ok": black_roundtrip_ok,
            "board_roundtrip_ok": board_roundtrip_ok,
        }

    def prepare_data(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
        raw_games = self.load_lichess_games_san()
        parsed_games = self.parse_games_for_player(raw_games)
        train_core_games, val_games, test_games = split_games_game_level(
            parsed_games,
            seed=self.config.random_seed,
            test_frac=self.config.test_frac,
            val_frac_within_train=self.config.val_frac_within_train,
        )

        bundles = {
            min_context_ply: self.build_bundle_for_min_context(
                train_core_games,
                val_games,
                test_games,
                min_context_ply,
            )
            for min_context_ply in self.config.min_context_candidates
        }
        sanity_checks = {
            str(min_context_ply): self.run_correctness_checks(bundle)
            for min_context_ply, bundle in bundles.items()
        }

        print("Loaded raw games:", len(raw_games))
        print("Parsed usable games:", len(parsed_games))
        print("Train games:", len(train_core_games), "| Val games:", len(val_games), "| Test games:", len(test_games))
        for min_context_ply, bundle in bundles.items():
            print(
                f"min_context={min_context_ply} | "
                f"train_examples={len(bundle['train_core_examples'])} | "
                f"val_examples={len(bundle['val_examples'])} | "
                f"test_examples={len(bundle['test_examples'])}"
            )
        print("Sanity checks:", sanity_checks)
        return parsed_games, bundles, sanity_checks

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, token=self.hf_token)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def load_base_model(self, tokenizer=None):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, token=self.hf_token)
        if tokenizer is not None and len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(self.device)
        model.eval()
        return model

    def build_lora_model(self, tokenizer, candidate_cfg: dict[str, Any]):
        base = AutoModelForCausalLM.from_pretrained(self.config.model_id, token=self.hf_token)
        if len(tokenizer) != base.get_input_embeddings().num_embeddings:
            base.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token_id is not None:
            base.config.pad_token_id = tokenizer.pad_token_id
        base = base.to(self.device)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(candidate_cfg["lora_rank"]),
            lora_alpha=int(candidate_cfg["lora_alpha"]),
            lora_dropout=float(candidate_cfg["lora_dropout"]),
            bias="none",
            target_modules=candidate_cfg["target_modules"],
        )
        return get_peft_model(base, lora_cfg)

    def build_sft_records(self, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [{"prompt": ex["prompt"], "target_text": " " + ex["norm_target"]} for ex in examples]

    def tokenize_sft_record(self, record: dict[str, str], tokenizer, max_length: int | None = None) -> dict[str, Any]:
        max_length = max_length or self.config.stage1_max_length
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(record["target_text"], add_special_tokens=False)["input_ids"]
        if not target_ids:
            raise ValueError("Target tokenization returned no ids.")

        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1]
        max_prompt_len = max_length - len(target_ids)
        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer requires a pad or eos token.")
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = ([-100] * len(prompt_ids)) + target_ids + ([-100] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def prepare_sft_dataset(self, examples: list[dict[str, Any]], tokenizer) -> Dataset:
        ds = Dataset.from_list(self.build_sft_records(examples))
        return ds.map(
            lambda row: self.tokenize_sft_record(row, tokenizer, max_length=self.config.stage1_max_length),
            remove_columns=ds.column_names,
        )

    def summarize_train_result(self, metrics: dict[str, Any]) -> dict[str, Any]:
        keep: dict[str, Any] = {}
        for key in ("train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "epoch"):
            if key in metrics:
                keep[key] = metrics[key]
        return keep

    def merge_metric_buckets(self, bucket_dicts: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        merged: defaultdict[str, Counter] = defaultdict(Counter)
        for bucket in bucket_dicts:
            for name, row in bucket.items():
                merged[name]["total"] += int(row["total"])
                merged[name]["top1_correct"] += int(row["top1_correct"])
                merged[name]["top5_correct"] += int(row["top5_correct"])

        out: dict[str, dict[str, Any]] = {}
        for name, counts in merged.items():
            total = counts["total"]
            out[name] = {
                "top1": (counts["top1_correct"] / total) if total else 0.0,
                "top5": (counts["top5_correct"] / total) if total else 0.0,
                "total": int(total),
                "top1_correct": int(counts["top1_correct"]),
                "top5_correct": int(counts["top5_correct"]),
            }
        return out

    def merge_eval_reports(self, reports: list[dict[str, Any]], debug_n: int = 10) -> dict[str, Any]:
        total = sum(report["total"] for report in reports)
        top1_correct = sum(report["top1_correct"] for report in reports)
        top5_correct = sum(report["top5_correct"] for report in reports)
        debug_rows: list[dict[str, Any]] = []
        for report in reports:
            debug_rows.extend(report.get("debug_rows", []))

        return {
            "top1": (top1_correct / total) if total else 0.0,
            "top5": (top5_correct / total) if total else 0.0,
            "total": total,
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "by_side": self.merge_metric_buckets([report.get("by_side", {}) for report in reports]),
            "by_phase": self.merge_metric_buckets([report.get("by_phase", {}) for report in reports]),
            "debug_rows": debug_rows[:debug_n],
        }

    def evaluate_model_bundle(
        self,
        model_bundle: dict[str, Any],
        tokenizer,
        eval_examples: list[dict[str, Any]],
        *,
        debug_n: int = 10,
    ) -> dict[str, Any]:
        if set(model_bundle.keys()) == {"unified"}:
            return evaluate_ranker(
                model_bundle["unified"],
                tokenizer,
                eval_examples,
                self.device,
                top_k=self.config.top_k_eval,
                debug_n=debug_n,
                max_context_moves=self.config.max_context_moves,
            )

        reports = []
        for side in ("white", "black"):
            side_examples = [ex for ex in eval_examples if ex["side"] == side]
            if not side_examples:
                continue
            if side not in model_bundle:
                raise ValueError(f"Missing side-specific model for {side}.")
            reports.append(
                evaluate_ranker(
                    model_bundle[side],
                    tokenizer,
                    side_examples,
                    self.device,
                    top_k=self.config.top_k_eval,
                    debug_n=debug_n,
                    max_context_moves=self.config.max_context_moves,
                )
            )
        return self.merge_eval_reports(reports, debug_n=debug_n)

    def train_sft_model(self, train_examples: list[dict[str, Any]], tokenizer, candidate_cfg: dict[str, Any], run_name: str):
        tokenized = self.prepare_sft_dataset(train_examples, tokenizer)
        model = self.build_lora_model(tokenizer, candidate_cfg)
        model.print_trainable_parameters()

        train_args = TrainingArguments(
            output_dir=str(self.output_dirs["scratch"] / run_name),
            per_device_train_batch_size=self.config.resolved_stage1_batch_size(),
            per_device_eval_batch_size=max(1, self.config.resolved_stage1_batch_size()),
            learning_rate=float(candidate_cfg["learning_rate"]),
            num_train_epochs=int(candidate_cfg["num_train_epochs"]),
            logging_steps=25,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized,
            data_collator=default_data_collator,
        )
        train_result = trainer.train()
        model.eval()
        return model, self.summarize_train_result(train_result.metrics)

    def cleanup_model_bundle(self, model_bundle: dict[str, Any]) -> None:
        self.cleanup_memory(*model_bundle.values())

    def run_stage1_candidate(
        self,
        train_examples: list[dict[str, Any]],
        eval_examples: list[dict[str, Any]],
        tokenizer,
        candidate_cfg: dict[str, Any],
        *,
        metric_prefix: str,
        run_name: str,
        debug_n: int = 8,
        save_dir: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if candidate_cfg["adapter_mode"] == "unified":
            model, train_summary = self.train_sft_model(train_examples, tokenizer, candidate_cfg, run_name)
            model_bundle = {"unified": model}
            train_summaries = {"unified": train_summary}
        else:
            model_bundle = {}
            train_summaries = {}
            for side in ("white", "black"):
                side_train = [ex for ex in train_examples if ex["side"] == side]
                side_eval = [ex for ex in eval_examples if ex["side"] == side]
                if not side_eval:
                    continue
                if not side_train:
                    raise ValueError(f"No training examples for side {side}.")
                side_model, train_summary = self.train_sft_model(
                    side_train,
                    tokenizer,
                    candidate_cfg,
                    f"{run_name}_{side}",
                )
                model_bundle[side] = side_model
                train_summaries[side] = train_summary

        report = self.evaluate_model_bundle(model_bundle, tokenizer, eval_examples, debug_n=debug_n)
        artifact_dir = self.save_model_bundle(model_bundle, tokenizer, save_dir) if save_dir is not None else None
        row = {
            **candidate_cfg,
            f"{metric_prefix}_top1": report["top1"],
            f"{metric_prefix}_top5": report["top5"],
            f"{metric_prefix}_total": report["total"],
            "trained_sides": sorted(model_bundle.keys()),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_summaries": train_summaries,
        }
        self.cleanup_model_bundle(model_bundle)
        return row, report, artifact_dir

    def score_candidate_scores_trainable(self, model, tokenizer, example: dict[str, Any], candidate_moves: list[str]) -> torch.Tensor:
        prompt_text = example["prompt"] if example.get("prompt") else build_prompt(
            example,
            max_context_moves=self.config.max_context_moves,
        )
        input_ids, attention_mask, prompt_lens = build_scoring_batch(tokenizer, prompt_text, candidate_moves)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_lens = prompt_lens.to(self.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()
        positions = torch.arange(shift_labels.shape[1], device=self.device).unsqueeze(0)
        candidate_mask = positions >= (prompt_lens.unsqueeze(1) - 1)
        valid_mask = shift_mask & candidate_mask

        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = token_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return (gathered * valid_mask).sum(dim=1)

    def build_reranker_records(
        self,
        examples: list[dict[str, Any]],
        baseline_model,
        tokenizer,
        *,
        seed: int,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        records: list[dict[str, Any]] = []
        for ex in examples:
            ranked = score_legal_moves(
                baseline_model,
                tokenizer,
                ex,
                self.device,
                max_context_moves=self.config.max_context_moves,
            )
            hard_negatives = pick_top_wrong_moves(
                ranked,
                ex["norm_target"],
                limit=self.config.reranker_hard_negatives,
            )
            negatives = select_reranker_negatives(
                ex,
                rng=rng,
                hard_negative_moves=hard_negatives,
                uniform_count=self.config.reranker_uniform_negatives,
                hard_count=self.config.reranker_hard_negatives,
            )
            if not negatives:
                continue
            candidate_moves = [ex["norm_target"], *negatives]
            rng.shuffle(candidate_moves)
            records.append(
                {
                    "example": ex,
                    "candidate_moves": candidate_moves,
                    "label": candidate_moves.index(ex["norm_target"]),
                    "hard_negatives": hard_negatives,
                }
            )
        return records

    def train_reranker_model(self, train_records: list[dict[str, Any]], tokenizer, candidate_cfg: dict[str, Any], run_name: str):
        model = self.build_lora_model(tokenizer, candidate_cfg)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(candidate_cfg["learning_rate"]),
        )

        history: list[dict[str, Any]] = []
        for epoch_idx in range(int(candidate_cfg["num_train_epochs"])):
            rng = random.Random(self.config.random_seed + epoch_idx)
            order = list(range(len(train_records)))
            rng.shuffle(order)
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            optimizer_steps = 0

            for local_step, record_idx in enumerate(order, start=1):
                record = train_records[record_idx]
                scores = self.score_candidate_scores_trainable(
                    model,
                    tokenizer,
                    record["example"],
                    record["candidate_moves"],
                )
                labels = torch.tensor([record["label"]], device=self.device, dtype=torch.long)
                loss = F.cross_entropy(scores.unsqueeze(0), labels)
                loss = loss / self.config.reranker_gradient_accumulation_steps
                loss.backward()
                running_loss += float(loss.item()) * self.config.reranker_gradient_accumulation_steps

                if (
                    local_step % self.config.reranker_gradient_accumulation_steps == 0
                    or local_step == len(order)
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1

            history.append(
                {
                    "epoch": epoch_idx + 1,
                    "avg_loss": running_loss / max(1, len(order)),
                    "optimizer_steps": optimizer_steps,
                }
            )
            print(
                f"{run_name} | epoch={epoch_idx + 1} | "
                f"avg_loss={history[-1]['avg_loss']:.4f} | optimizer_steps={optimizer_steps}"
            )

        model.eval()
        return model, history

    def run_stage2_candidate(
        self,
        train_examples: list[dict[str, Any]],
        eval_examples: list[dict[str, Any]],
        tokenizer,
        baseline_model,
        candidate_cfg: dict[str, Any],
        *,
        metric_prefix: str,
        run_name: str,
        debug_n: int = 8,
        save_dir: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if candidate_cfg["adapter_mode"] == "unified":
            train_records = self.build_reranker_records(
                train_examples,
                baseline_model,
                tokenizer,
                seed=self.config.random_seed,
            )
            model, train_history = self.train_reranker_model(train_records, tokenizer, candidate_cfg, run_name)
            model_bundle = {"unified": model}
            train_summaries = {"unified": train_history}
        else:
            model_bundle = {}
            train_summaries = {}
            for side in ("white", "black"):
                side_train = [ex for ex in train_examples if ex["side"] == side]
                side_eval = [ex for ex in eval_examples if ex["side"] == side]
                if not side_eval:
                    continue
                if not side_train:
                    raise ValueError(f"No training examples for side {side}.")
                side_records = self.build_reranker_records(
                    side_train,
                    baseline_model,
                    tokenizer,
                    seed=self.config.random_seed + (0 if side == "white" else 1),
                )
                side_model, train_history = self.train_reranker_model(
                    side_records,
                    tokenizer,
                    candidate_cfg,
                    f"{run_name}_{side}",
                )
                model_bundle[side] = side_model
                train_summaries[side] = train_history

        report = self.evaluate_model_bundle(model_bundle, tokenizer, eval_examples, debug_n=debug_n)
        artifact_dir = self.save_model_bundle(model_bundle, tokenizer, save_dir) if save_dir is not None else None
        row = {
            **candidate_cfg,
            f"{metric_prefix}_top1": report["top1"],
            f"{metric_prefix}_top5": report["top5"],
            f"{metric_prefix}_total": report["total"],
            "trained_sides": sorted(model_bundle.keys()),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_summaries": train_summaries,
        }
        self.cleanup_model_bundle(model_bundle)
        return row, report, artifact_dir

    def run(self) -> dict[str, Any]:
        self.seed_everything(self.config.random_seed)
        self.hf_token = self.maybe_login_hf()
        self.print_header()

        parsed_games, bundles, sanity_checks = self.prepare_data()
        tokenizer = self.load_tokenizer()

        baseline_model = self.load_base_model(tokenizer)
        baseline_val_reports = {}
        for min_context_ply, bundle in bundles.items():
            report = evaluate_ranker(
                baseline_model,
                tokenizer,
                bundle["val_examples"],
                self.device,
                top_k=self.config.top_k_eval,
                debug_n=6,
                max_context_moves=self.config.max_context_moves,
            )
            baseline_val_reports[min_context_ply] = report
            print(
                f"Baseline val | min_context={min_context_ply} | "
                f"Top-1={report['top1']:.4f} ({report['top1_correct']}/{report['total']}) | "
                f"Top-5={report['top5']:.4f} ({report['top5_correct']}/{report['total']})"
            )

        determinism_examples = bundles[self.config.min_context_candidates[0]]["val_examples"][:5]
        deterministic_a = evaluate_ranker(
            baseline_model,
            tokenizer,
            determinism_examples,
            self.device,
            top_k=self.config.top_k_eval,
            debug_n=2,
            max_context_moves=self.config.max_context_moves,
        )
        deterministic_b = evaluate_ranker(
            baseline_model,
            tokenizer,
            determinism_examples,
            self.device,
            top_k=self.config.top_k_eval,
            debug_n=2,
            max_context_moves=self.config.max_context_moves,
        )
        determinism_check = (
            deterministic_a["top1"] == deterministic_b["top1"]
            and deterministic_a["top5"] == deterministic_b["top5"]
            and deterministic_a["debug_rows"][:2] == deterministic_b["debug_rows"][:2]
        )
        print("Determinism check on repeated baseline evaluation:", determinism_check)
        self.cleanup_memory(baseline_model)

        stage1_search_rows = []
        for min_context_ply in self.config.min_context_candidates:
            bundle = bundles[min_context_ply]
            for num_train_epochs in self.config.stage1_first_sweep_epochs:
                for learning_rate in self.config.stage1_first_sweep_lrs:
                    candidate_cfg = {
                        "stage": "stage1_sft",
                        "adapter_mode": "unified",
                        "min_context_ply": min_context_ply,
                        "learning_rate": learning_rate,
                        "num_train_epochs": num_train_epochs,
                        "lora_rank": 16,
                        "lora_alpha": self.config.stage1_lora_alpha,
                        "lora_dropout": self.config.stage1_lora_dropout,
                        "target_modules": self.config.stage1_target_modules,
                    }
                    row, _, _ = self.run_stage1_candidate(
                        bundle["train_core_examples"],
                        bundle["val_examples"],
                        tokenizer,
                        candidate_cfg,
                        metric_prefix="val",
                        run_name=f"stage1_search_mc{min_context_ply}_ep{num_train_epochs}_lr{learning_rate}",
                        debug_n=4,
                    )
                    stage1_search_rows.append(row)
                    print(
                        f"Stage 1 search | min_context={min_context_ply} | epochs={num_train_epochs} | "
                        f"lr={learning_rate:.0e} | Top-1={row['val_top1']:.4f} | Top-5={row['val_top5']:.4f}"
                    )

        best_stage1_search = self.sort_metric_rows(stage1_search_rows, metric_prefix="val")[0]
        selected_min_context = int(best_stage1_search["min_context_ply"])
        selected_bundle = bundles[selected_min_context]
        selected_baseline_val = baseline_val_reports[selected_min_context]

        print("Selected min_context_ply from Stage 1 search:", selected_min_context)
        print("Selected Stage 1 search row:", best_stage1_search)
        print(
            "Baseline threshold for promotion:",
            selected_baseline_val["top1"],
            "->",
            selected_baseline_val["top1"] * self.config.metric_goal_relative_top1,
        )

        stage1_refine_rows = []
        for adapter_mode in ("unified", "per_side"):
            for lora_rank in self.config.stage1_refine_ranks:
                candidate_cfg = {
                    "stage": "stage1_sft",
                    "adapter_mode": adapter_mode,
                    "min_context_ply": selected_min_context,
                    "learning_rate": best_stage1_search["learning_rate"],
                    "num_train_epochs": best_stage1_search["num_train_epochs"],
                    "lora_rank": lora_rank,
                    "lora_alpha": self.config.stage1_lora_alpha,
                    "lora_dropout": self.config.stage1_lora_dropout,
                    "target_modules": self.config.stage1_target_modules,
                }
                row, _, _ = self.run_stage1_candidate(
                    selected_bundle["train_core_examples"],
                    selected_bundle["val_examples"],
                    tokenizer,
                    candidate_cfg,
                    metric_prefix="val",
                    run_name=f"stage1_refine_{adapter_mode}_r{lora_rank}",
                    debug_n=5,
                )
                stage1_refine_rows.append(row)
                print(
                    f"Stage 1 refine | mode={adapter_mode} | rank={lora_rank} | "
                    f"Top-1={row['val_top1']:.4f} | Top-5={row['val_top5']:.4f}"
                )

        best_stage1_val = self.sort_metric_rows(stage1_refine_rows, metric_prefix="val")[0]
        print("Best Stage 1 validation candidate:", best_stage1_val)

        stage2_triggered = best_stage1_val["val_top1"] < (
            selected_baseline_val["top1"] * self.config.metric_goal_relative_top1
        )
        stage2_val = None
        stage2_val_report = None
        if stage2_triggered:
            print("Stage 2 triggered because Stage 1 missed the validation promotion threshold.")
            baseline_model = self.load_base_model(tokenizer)
            stage2_cfg = {
                "stage": "stage2_reranker",
                "adapter_mode": best_stage1_val["adapter_mode"],
                "min_context_ply": selected_min_context,
                "learning_rate": best_stage1_val["learning_rate"],
                "num_train_epochs": best_stage1_val["num_train_epochs"],
                "lora_rank": best_stage1_val["lora_rank"],
                "lora_alpha": self.config.stage1_lora_alpha,
                "lora_dropout": self.config.stage1_lora_dropout,
                "target_modules": self.config.stage1_target_modules,
            }
            stage2_val, stage2_val_report, _ = self.run_stage2_candidate(
                selected_bundle["train_core_examples"],
                selected_bundle["val_examples"],
                tokenizer,
                baseline_model,
                stage2_cfg,
                metric_prefix="val",
                run_name="stage2_val",
                debug_n=6,
            )
            self.cleanup_memory(baseline_model)
            print("Stage 2 validation candidate:", stage2_val)
        else:
            print("Stage 2 skipped because Stage 1 already met the validation target.")

        best_val_candidate = best_stage1_val
        if stage2_val is not None:
            best_val_candidate = self.sort_metric_rows([best_stage1_val, stage2_val], metric_prefix="val")[0]
        print("Selected best validation candidate:", best_val_candidate)

        baseline_model = self.load_base_model(tokenizer)
        baseline_test_report = evaluate_ranker(
            baseline_model,
            tokenizer,
            selected_bundle["test_examples"],
            self.device,
            top_k=self.config.top_k_eval,
            debug_n=10,
            max_context_moves=self.config.max_context_moves,
        )
        self.cleanup_memory(baseline_model)

        if best_val_candidate["stage"] == "stage2_reranker":
            baseline_model = self.load_base_model(tokenizer)
            best_test_row, best_test_report, best_model_artifact = self.run_stage2_candidate(
                selected_bundle["train_full_examples"],
                selected_bundle["test_examples"],
                tokenizer,
                baseline_model,
                best_val_candidate,
                metric_prefix="test",
                run_name="stage2_final",
                debug_n=10,
                save_dir=self.output_dirs["models"] / "best_stage2",
            )
            self.cleanup_memory(baseline_model)
        else:
            best_test_row, best_test_report, best_model_artifact = self.run_stage1_candidate(
                selected_bundle["train_full_examples"],
                selected_bundle["test_examples"],
                tokenizer,
                best_val_candidate,
                metric_prefix="test",
                run_name="stage1_final",
                debug_n=10,
                save_dir=self.output_dirs["models"] / "best_stage1",
            )

        target_threshold = baseline_test_report["top1"] * self.config.metric_goal_relative_top1
        achieved_target = best_test_report["top1"] >= target_threshold

        print(
            f"Baseline test | Top-1={baseline_test_report['top1']:.4f} ({baseline_test_report['top1_correct']}/{baseline_test_report['total']}) | "
            f"Top-5={baseline_test_report['top5']:.4f} ({baseline_test_report['top5_correct']}/{baseline_test_report['total']})"
        )
        print(
            f"Best test | Top-1={best_test_report['top1']:.4f} ({best_test_report['top1_correct']}/{best_test_report['total']}) | "
            f"Top-5={best_test_report['top5']:.4f} ({best_test_report['top5_correct']}/{best_test_report['total']})"
        )
        print("Required Top-1 threshold:", target_threshold)
        print("Achieved +10% relative Top-1 target:", achieved_target)

        experiment_log = {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "approach": "lora_legal_move_ranking",
            "metric_goal": {
                "name": "relative_top1_improvement",
                "multiplier": self.config.metric_goal_relative_top1,
                "description": "Best model must reach test_top1 >= baseline_top1 * 1.10.",
            },
            "config": self.to_builtin(self.config.__dict__) | {
                "device": self.device,
                "selected_min_context_ply": selected_min_context,
            },
            "split": {
                "parsed_games": len(parsed_games),
                "train_core_games": len(selected_bundle["train_core_games"]),
                "val_games": len(selected_bundle["val_games"]),
                "train_full_games": len(selected_bundle["train_full_games"]),
                "test_games": len(selected_bundle["test_games"]),
                "train_core_examples": len(selected_bundle["train_core_examples"]),
                "val_examples": len(selected_bundle["val_examples"]),
                "train_full_examples": len(selected_bundle["train_full_examples"]),
                "test_examples": len(selected_bundle["test_examples"]),
                "test_by_side": dict(Counter(ex["side"] for ex in selected_bundle["test_examples"])),
                "sanity_checks": sanity_checks,
                "determinism_check": determinism_check,
            },
            "baseline": {
                "val_by_min_context": baseline_val_reports,
                "test": baseline_test_report,
            },
            "stage1": {
                "search_rows": stage1_search_rows,
                "refine_rows": stage1_refine_rows,
                "best_val": best_stage1_val,
            },
            "stage2": {
                "triggered": stage2_triggered,
                "best_val": stage2_val,
                "best_val_report": stage2_val_report,
            },
            "best_val": {
                "candidate": best_val_candidate,
                "source": best_val_candidate["stage"],
            },
            "best_test": {
                "candidate": best_test_row,
                "report": best_test_report,
                "meets_goal": achieved_target,
                "required_top1_threshold": target_threshold,
            },
            "by_side": {
                "baseline_test": baseline_test_report["by_side"],
                "best_test": best_test_report["by_side"],
            },
            "by_phase": {
                "baseline_test": baseline_test_report["by_phase"],
                "best_test": best_test_report["by_phase"],
            },
            "artifacts": {
                "best_model_dir": best_model_artifact,
                "output_root": str(self.output_dirs["root"]),
                "experiment_log": str(
                    self.output_dirs["logs"]
                    / f"{self.config.username}_{self.config.perf_type}_model3_{self.run_id}.json"
                ),
            },
        }
        experiment_log_path = self.write_experiment_log(experiment_log)
        print("Best model artifact:", best_model_artifact)
        print("Experiment log:", experiment_log_path)
        return experiment_log


def run_model3_experiment(config: Model3ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or Model3ExperimentConfig()
    experiment = Model3Experiment(config)
    return experiment.run()
