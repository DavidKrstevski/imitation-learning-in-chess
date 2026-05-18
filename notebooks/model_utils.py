from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from test_result_utils import (
    average_kl_divergence,
    build_aggregate_metrics,
    build_examples_from_games,
    build_result_plots,
    cleanup_torch_objects,
    ensure_player_dirs,
    ensure_root_dirs,
    evaluate_policy_model,
    load_base_model,
    load_lichess_games_san,
    load_lichess_user_profile,
    load_tokenizer,
    make_run_id,
    maybe_login_hf,
    min_total_games_required,
    parse_target_games,
    save_player_model,
    seed_everything,
    select_fixed_train_games,
    split_games_train_val_test,
    summarize_top_metrics,
    train_lora_model,
    write_json,
)


@dataclass
class ExperimentConfig:
    player_usernames: tuple[str, ...] = (
        "Vlad_Lazarev79",
        "ChessTheory64",
        "RubiRedhead",
        "UniversalRuler",
    )
    perf_type: str = "classical"
    max_games: int = 400
    rated_only: bool = False
    split_seed: int = 42
    split_strategy: str = "chronological"
    test_frac: float = 0.2
    val_frac_within_train: float = 0.2
    train_games_per_player: int = 150
    min_context_candidates: tuple[int, ...] = (0, 4)
    contexts_to_keep_from_baseline: int = 1
    model_id: str = "daavidhauser/chess-bot-3000-250m"
    max_length: int = 256
    candidate_scoring_batch_size: int = 64
    top_ks: tuple[int, ...] = (1, 3, 5, 10)
    debug_examples: int = 10
    per_device_train_batch_size: int = 4 if torch.cuda.is_available() else 1
    per_device_eval_batch_size: int = 4 if torch.cuda.is_available() else 1
    logging_steps: int = 25
    stage_a_learning_rates: tuple[float, ...] = (1.5e-4, 2e-4, 2.5e-4)
    stage_a_epochs: tuple[int, ...] = (1,)
    stage_a_lora_rank: int = 32
    stage_a_lora_alpha: int = 64
    stage_a_lora_dropout: float = 0.05
    stage_a_target_modules: str = "all-linear"
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    weight_decay: float = 0.01
    results_root_name: str = "results"
    run_id_override: str | None = None


def stage_b_lora_candidates() -> list[dict[str, Any]]:
    return [
        {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"},
        {"lora_rank": 24, "lora_alpha": 48, "lora_dropout": 0.05, "target_modules": "all-linear"},
        {"lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "target_modules": "all-linear"},
        {"lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.10, "target_modules": "all-linear"},
        {"lora_rank": 48, "lora_alpha": 96, "lora_dropout": 0.05, "target_modules": "all-linear"},
    ]


def ensure_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["mean_delta_top1_accuracy"]),
        float(row["mean_delta_top3_accuracy"]),
        float(row["mean_delta_top5_accuracy"]),
        float(row["mean_delta_top10_accuracy"]),
        -float(row["mean_delta_mean_rank"]),
        -float(row.get("mean_train_loss", 0.0)),
    )


def context_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["mean_top1_accuracy"]),
        float(row["mean_top3_accuracy"]),
        float(row["mean_top5_accuracy"]),
        float(row["mean_top10_accuracy"]),
        -float(row["mean_mean_rank"]),
    )


def sort_rows(rows: list[dict[str, Any]], key_func) -> list[dict[str, Any]]:
    return sorted(rows, key=key_func, reverse=True)


def build_run_name(config: ExperimentConfig, run_id: str) -> str:
    return f"multi_{len(config.player_usernames)}players_{config.perf_type}_{run_id}"


def summarize_player_dataset(player_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": player_data["username"],
        "elo": player_data["profile"]["elo"],
        "title": player_data["profile"]["title"],
        "perf_type": player_data["perf_type"],
        "raw_games_loaded": player_data["raw_games_loaded"],
        "parsed_games_used": player_data["parsed_games_used"],
        "split_strategy": player_data["split_strategy"],
        "train_games_available": player_data["train_games_available"],
        "train_games_used": player_data["train_games_used"],
        "val_games": player_data["val_games"],
        "test_games": player_data["test_games"],
        "example_counts_by_min_context": player_data["example_counts_by_min_context"],
    }


def prepare_player_data(
    config: ExperimentConfig,
    root_dirs: dict[str, Path],
    username: str,
) -> dict[str, Any]:
    player_dirs = ensure_player_dirs(root_dirs, username)
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
            f"to train on {config.train_games_per_player} games after the split."
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
    )

    bundles: dict[int, dict[str, Any]] = {}
    example_counts_by_min_context: dict[int, dict[str, int]] = {}
    for min_context in config.min_context_candidates:
        train_examples = build_examples_from_games(selected_train_games, min_context)
        val_examples = build_examples_from_games(val_games, min_context)
        test_examples = build_examples_from_games(test_games, min_context)
        train_val_examples = build_examples_from_games(selected_train_games + val_games, min_context)
        bundles[min_context] = {
            "train_examples": train_examples,
            "val_examples": val_examples,
            "test_examples": test_examples,
            "train_val_examples": train_val_examples,
        }
        example_counts_by_min_context[min_context] = {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "train_val_examples": len(train_val_examples),
        }

    player_data = {
        "username": username,
        "perf_type": config.perf_type,
        "profile": profile,
        "player_dirs": player_dirs,
        "raw_games_loaded": len(raw_games),
        "parsed_games_used": len(parsed_games),
        "split_strategy": config.split_strategy,
        "train_games_available": len(train_games),
        "train_games_used": len(selected_train_games),
        "val_games": len(val_games),
        "test_games": len(test_games),
        "example_counts_by_min_context": example_counts_by_min_context,
        "bundles": bundles,
    }
    write_json(player_dirs["root"] / "dataset_summary.json", summarize_player_dataset(player_data))
    return player_data


def evaluate_baseline_contexts(
    config: ExperimentConfig,
    tokenizer,
    player_data: dict[str, Any],
    device: str,
    hf_token: str | None,
) -> dict[int, dict[str, Any]]:
    baseline_model = load_base_model(config.model_id, tokenizer, device, hf_token)
    context_rows: dict[int, dict[str, Any]] = {}
    for min_context, bundle in player_data["bundles"].items():
        metrics, _ = evaluate_policy_model(
            model=baseline_model,
            tokenizer=tokenizer,
            examples=bundle["val_examples"],
            device=device,
            top_ks=config.top_ks,
            debug_n=config.debug_examples,
            candidate_scoring_batch_size=config.candidate_scoring_batch_size,
            max_length=config.max_length,
            return_distributions=False,
        )
        context_rows[min_context] = metrics
        print(
            f"baseline | {player_data['username']} | min_context={min_context} |",
            summarize_top_metrics(metrics, config.top_ks),
        )

    cleanup_torch_objects(baseline_model)
    del baseline_model
    return context_rows


def aggregate_baseline_contexts(
    config: ExperimentConfig,
    players: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for min_context in config.min_context_candidates:
        player_rows = []
        for player in players:
            metrics = player["baseline_val_by_context"][min_context]
            player_rows.append(
                {
                    "username": player["username"],
                    "elo": player["profile"]["elo"],
                    "val_examples": len(player["bundles"][min_context]["val_examples"]),
                    "top1_accuracy": metrics["top1_accuracy"],
                    "top3_accuracy": metrics["top3_accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                    "top10_accuracy": metrics["top10_accuracy"],
                    "mean_rank": metrics["mean_rank"],
                }
            )
        rows.append(
            {
                "min_context_ply": min_context,
                "player_count": len(player_rows),
                "mean_top1_accuracy": sum(row["top1_accuracy"] for row in player_rows) / len(player_rows),
                "mean_top3_accuracy": sum(row["top3_accuracy"] for row in player_rows) / len(player_rows),
                "mean_top5_accuracy": sum(row["top5_accuracy"] for row in player_rows) / len(player_rows),
                "mean_top10_accuracy": sum(row["top10_accuracy"] for row in player_rows) / len(player_rows),
                "mean_mean_rank": sum(row["mean_rank"] for row in player_rows) / len(player_rows),
                "players": player_rows,
            }
        )
    return sort_rows(rows, context_sort_key)


def make_candidate_config(
    config: ExperimentConfig,
    learning_rate: float,
    num_train_epochs: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
) -> ExperimentConfig:
    return replace(
        config,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )


def evaluate_tuning_candidate(
    config: ExperimentConfig,
    tokenizer,
    players: list[dict[str, Any]],
    min_context_ply: int,
    learning_rate: float,
    num_train_epochs: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    device: str,
    hf_token: str | None,
    stage_name: str,
) -> dict[str, Any]:
    candidate_name = (
        f"{stage_name}_mc{min_context_ply}_"
        f"lr{learning_rate:.0e}_ep{num_train_epochs}_"
        f"r{lora_rank}_a{lora_alpha}_d{str(lora_dropout).replace('.', '')}"
    )
    candidate_config = make_candidate_config(
        config,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    player_rows = []
    for player in players:
        bundle = player["bundles"][min_context_ply]
        model, train_summary = train_lora_model(
            config=candidate_config,
            tokenizer=tokenizer,
            train_examples=bundle["train_examples"],
            model_id=config.model_id,
            player_dirs=player["player_dirs"],
            run_name=candidate_name,
            device=device,
            hf_token=hf_token,
        )
        val_metrics, _ = evaluate_policy_model(
            model=model,
            tokenizer=tokenizer,
            examples=bundle["val_examples"],
            device=device,
            top_ks=config.top_ks,
            debug_n=config.debug_examples,
            candidate_scoring_batch_size=config.candidate_scoring_batch_size,
            max_length=config.max_length,
            return_distributions=False,
        )
        baseline_metrics = player["baseline_val_by_context"][min_context_ply]
        player_rows.append(
            {
                "username": player["username"],
                "elo": player["profile"]["elo"],
                "val_examples": len(bundle["val_examples"]),
                "baseline_top1_accuracy": baseline_metrics["top1_accuracy"],
                "baseline_top3_accuracy": baseline_metrics["top3_accuracy"],
                "baseline_top5_accuracy": baseline_metrics["top5_accuracy"],
                "baseline_top10_accuracy": baseline_metrics["top10_accuracy"],
                "baseline_mean_rank": baseline_metrics["mean_rank"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_top3_accuracy": val_metrics["top3_accuracy"],
                "val_top5_accuracy": val_metrics["top5_accuracy"],
                "val_top10_accuracy": val_metrics["top10_accuracy"],
                "val_mean_rank": val_metrics["mean_rank"],
                "delta_top1_accuracy": val_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"],
                "delta_top3_accuracy": val_metrics["top3_accuracy"] - baseline_metrics["top3_accuracy"],
                "delta_top5_accuracy": val_metrics["top5_accuracy"] - baseline_metrics["top5_accuracy"],
                "delta_top10_accuracy": val_metrics["top10_accuracy"] - baseline_metrics["top10_accuracy"],
                "delta_mean_rank": val_metrics["mean_rank"] - baseline_metrics["mean_rank"],
                "train_loss": float((train_summary["train_result_metrics"] or {}).get("train_loss", 0.0)),
                "train_runtime": float((train_summary["train_result_metrics"] or {}).get("train_runtime", 0.0)),
            }
        )
        print(candidate_name, "|", player["username"], summarize_top_metrics(val_metrics, config.top_ks))
        cleanup_torch_objects(model)
        del model

    return {
        "candidate_name": candidate_name,
        "stage": stage_name,
        "min_context_ply": min_context_ply,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "player_count": len(player_rows),
        "mean_val_top1_accuracy": sum(row["val_top1_accuracy"] for row in player_rows) / len(player_rows),
        "mean_val_top3_accuracy": sum(row["val_top3_accuracy"] for row in player_rows) / len(player_rows),
        "mean_val_top5_accuracy": sum(row["val_top5_accuracy"] for row in player_rows) / len(player_rows),
        "mean_val_top10_accuracy": sum(row["val_top10_accuracy"] for row in player_rows) / len(player_rows),
        "mean_val_mean_rank": sum(row["val_mean_rank"] for row in player_rows) / len(player_rows),
        "mean_delta_top1_accuracy": sum(row["delta_top1_accuracy"] for row in player_rows) / len(player_rows),
        "mean_delta_top3_accuracy": sum(row["delta_top3_accuracy"] for row in player_rows) / len(player_rows),
        "mean_delta_top5_accuracy": sum(row["delta_top5_accuracy"] for row in player_rows) / len(player_rows),
        "mean_delta_top10_accuracy": sum(row["delta_top10_accuracy"] for row in player_rows) / len(player_rows),
        "mean_delta_mean_rank": sum(row["delta_mean_rank"] for row in player_rows) / len(player_rows),
        "mean_train_loss": sum(row["train_loss"] for row in player_rows) / len(player_rows),
        "mean_train_runtime": sum(row["train_runtime"] for row in player_rows) / len(player_rows),
        "players": player_rows,
    }


def run_stage_a(
    config: ExperimentConfig,
    tokenizer,
    players: list[dict[str, Any]],
    candidate_contexts: list[int],
    root_dirs: dict[str, Path],
    device: str,
    hf_token: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for min_context in candidate_contexts:
        for learning_rate in config.stage_a_learning_rates:
            for num_train_epochs in config.stage_a_epochs:
                row = evaluate_tuning_candidate(
                    config=config,
                    tokenizer=tokenizer,
                    players=players,
                    min_context_ply=min_context,
                    learning_rate=learning_rate,
                    num_train_epochs=num_train_epochs,
                    lora_rank=config.stage_a_lora_rank,
                    lora_alpha=config.stage_a_lora_alpha,
                    lora_dropout=config.stage_a_lora_dropout,
                    target_modules=config.stage_a_target_modules,
                    device=device,
                    hf_token=hf_token,
                    stage_name="stage_a",
                )
                rows.append(row)
                sorted_rows = sort_rows(rows, candidate_sort_key)
                write_json(root_dirs["root"] / "tuning_stage_a.json", {"rows": sorted_rows, "best": sorted_rows[0]})
    sorted_rows = sort_rows(rows, candidate_sort_key)
    return sorted_rows, sorted_rows[0]


def run_stage_b(
    config: ExperimentConfig,
    tokenizer,
    players: list[dict[str, Any]],
    best_stage_a: dict[str, Any],
    root_dirs: dict[str, Path],
    device: str,
    hf_token: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for lora_candidate in stage_b_lora_candidates():
        row = evaluate_tuning_candidate(
            config=config,
            tokenizer=tokenizer,
            players=players,
            min_context_ply=int(best_stage_a["min_context_ply"]),
            learning_rate=float(best_stage_a["learning_rate"]),
            num_train_epochs=int(best_stage_a["num_train_epochs"]),
            lora_rank=int(lora_candidate["lora_rank"]),
            lora_alpha=int(lora_candidate["lora_alpha"]),
            lora_dropout=float(lora_candidate["lora_dropout"]),
            target_modules=str(lora_candidate["target_modules"]),
            device=device,
            hf_token=hf_token,
            stage_name="stage_b",
        )
        rows.append(row)
        sorted_rows = sort_rows(rows, candidate_sort_key)
        write_json(root_dirs["root"] / "tuning_stage_b.json", {"rows": sorted_rows, "best": sorted_rows[0]})
    sorted_rows = sort_rows(rows, candidate_sort_key)
    return sorted_rows, sorted_rows[0]


def run_final_evaluation(
    config: ExperimentConfig,
    tokenizer,
    players: list[dict[str, Any]],
    best_stage_a: dict[str, Any],
    best_stage_b: dict[str, Any],
    root_dirs: dict[str, Path],
    device: str,
    hf_token: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful_rows = []
    failed_rows = []
    final_config = make_candidate_config(
        config,
        learning_rate=float(best_stage_b["learning_rate"]),
        num_train_epochs=int(best_stage_b["num_train_epochs"]),
        lora_rank=int(best_stage_b["lora_rank"]),
        lora_alpha=int(best_stage_b["lora_alpha"]),
        lora_dropout=float(best_stage_b["lora_dropout"]),
        target_modules=str(best_stage_b["target_modules"]),
    )
    selected_min_context = int(best_stage_a["min_context_ply"])

    for player in players:
        try:
            bundle = player["bundles"][selected_min_context]
            baseline_model = load_base_model(config.model_id, tokenizer, device, hf_token)
            baseline_test_metrics, baseline_test_distributions = evaluate_policy_model(
                model=baseline_model,
                tokenizer=tokenizer,
                examples=bundle["test_examples"],
                device=device,
                top_ks=config.top_ks,
                debug_n=config.debug_examples,
                candidate_scoring_batch_size=config.candidate_scoring_batch_size,
                max_length=config.max_length,
                return_distributions=True,
            )
            cleanup_torch_objects(baseline_model)
            del baseline_model

            final_model, final_train_summary = train_lora_model(
                config=final_config,
                tokenizer=tokenizer,
                train_examples=bundle["train_val_examples"],
                model_id=config.model_id,
                player_dirs=player["player_dirs"],
                run_name="final",
                device=device,
                hf_token=hf_token,
            )
            write_json(player["player_dirs"]["root"] / "trainer_logs.json", final_train_summary)

            finetuned_test_metrics, finetuned_test_distributions = evaluate_policy_model(
                model=final_model,
                tokenizer=tokenizer,
                examples=bundle["test_examples"],
                device=device,
                top_ks=config.top_ks,
                debug_n=config.debug_examples,
                candidate_scoring_batch_size=config.candidate_scoring_batch_size,
                max_length=config.max_length,
                return_distributions=True,
            )

            average_kl = average_kl_divergence(
                finetuned_distributions=finetuned_test_distributions or [],
                baseline_distributions=baseline_test_distributions or [],
            )
            comparison = {
                "delta_top1_accuracy": finetuned_test_metrics["top1_accuracy"] - baseline_test_metrics["top1_accuracy"],
                "delta_top3_accuracy": finetuned_test_metrics["top3_accuracy"] - baseline_test_metrics["top3_accuracy"],
                "delta_top5_accuracy": finetuned_test_metrics["top5_accuracy"] - baseline_test_metrics["top5_accuracy"],
                "delta_top10_accuracy": finetuned_test_metrics["top10_accuracy"] - baseline_test_metrics["top10_accuracy"],
                "delta_mean_rank": finetuned_test_metrics["mean_rank"] - baseline_test_metrics["mean_rank"],
                "average_kl_finetuned_vs_baseline": average_kl,
            }

            write_json(
                player["player_dirs"]["root"] / "baseline_metrics.json",
                {
                    "validation_by_min_context": player["baseline_val_by_context"],
                    "selected_min_context_ply": selected_min_context,
                    "selected_test_baseline": baseline_test_metrics,
                },
            )

            final_report = {
                "run_id": root_dirs["root"].name,
                "username": player["username"],
                "model_id": config.model_id,
                "profile": player["profile"],
                "config": asdict(config),
                "selected_min_context_ply": selected_min_context,
                "best_stage_a": best_stage_a,
                "best_stage_b": best_stage_b,
                "dataset_summary": summarize_player_dataset(player),
                "baseline_test_metrics": baseline_test_metrics,
                "finetuned_test_metrics": finetuned_test_metrics,
                "comparison": comparison,
                "training": final_train_summary,
            }
            write_json(player["player_dirs"]["root"] / "final_report.json", final_report)
            save_player_model(
                config=final_config,
                model=final_model,
                tokenizer=tokenizer,
                player_dirs=player["player_dirs"],
                player_summary=final_report,
            )

            successful_rows.append(
                {
                    "username": player["username"],
                    "elo": player["profile"]["elo"],
                    "profile_games": player["profile"]["games"],
                    "parsed_games_used": player["parsed_games_used"],
                    "train_games_used": player["train_games_used"],
                    "val_games": player["val_games"],
                    "test_games": player["test_games"],
                    "train_examples": len(bundle["train_val_examples"]),
                    "test_examples": len(bundle["test_examples"]),
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
                    "average_kl_finetuned_vs_baseline": average_kl,
                    "player_results_dir": str(player["player_dirs"]["root"]),
                }
            )
            print("final |", player["username"], summarize_top_metrics(finetuned_test_metrics, config.top_ks))
            cleanup_torch_objects(final_model)
            del final_model
        except Exception as exc:
            failed_rows.append(
                {
                    "username": player["username"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            write_json(player["player_dirs"]["root"] / "error.json", failed_rows[-1])

    return successful_rows, failed_rows


def run_multi_player_model_search(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    config = replace(
        config,
        player_usernames=tuple(config.player_usernames) if not isinstance(config.player_usernames, str) else (config.player_usernames,),
        min_context_candidates=ensure_tuple(config.min_context_candidates),
        stage_a_learning_rates=ensure_tuple(config.stage_a_learning_rates),
        stage_a_epochs=ensure_tuple(config.stage_a_epochs),
    )
    if not config.player_usernames:
        raise ValueError("Provide at least one player username.")

    run_id = config.run_id_override or make_run_id()
    run_name = build_run_name(config, run_id)
    root_dirs = ensure_root_dirs(run_name, config.results_root_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = maybe_login_hf()
    seed_everything(config.split_seed)
    write_json(root_dirs["root"] / "config.json", asdict(config))

    print("Run ID:", run_name)
    print("Device:", device)
    print("Results root:", root_dirs["root"])
    print("Players:", config.player_usernames)
    print("Min total games required per player:", min_total_games_required(config))

    tokenizer = load_tokenizer(config.model_id, hf_token)

    players = []
    for username in config.player_usernames:
        print("=" * 100)
        print(f"Preparing player: {username}")
        player_data = prepare_player_data(config, root_dirs, username)
        player_data["baseline_val_by_context"] = evaluate_baseline_contexts(
            config=config,
            tokenizer=tokenizer,
            player_data=player_data,
            device=device,
            hf_token=hf_token,
        )
        players.append(player_data)

    baseline_context_rows = aggregate_baseline_contexts(config, players)
    selected_contexts = [
        int(row["min_context_ply"])
        for row in baseline_context_rows[: max(1, min(config.contexts_to_keep_from_baseline, len(baseline_context_rows)))]
    ]
    baseline_summary = {
        "rows": baseline_context_rows,
        "selected_context_candidates": selected_contexts,
    }
    write_json(root_dirs["root"] / "baseline_validation.json", baseline_summary)

    stage_a_rows, best_stage_a = run_stage_a(
        config=config,
        tokenizer=tokenizer,
        players=players,
        candidate_contexts=selected_contexts,
        root_dirs=root_dirs,
        device=device,
        hf_token=hf_token,
    )
    stage_b_rows, best_stage_b = run_stage_b(
        config=config,
        tokenizer=tokenizer,
        players=players,
        best_stage_a=best_stage_a,
        root_dirs=root_dirs,
        device=device,
        hf_token=hf_token,
    )

    successful_rows, failed_rows = run_final_evaluation(
        config=config,
        tokenizer=tokenizer,
        players=players,
        best_stage_a=best_stage_a,
        best_stage_b=best_stage_b,
        root_dirs=root_dirs,
        device=device,
        hf_token=hf_token,
    )
    successful_rows = sorted(
        successful_rows,
        key=lambda row: (row["elo"] is None, row["elo"] if row["elo"] is not None else 10**9, row["username"].lower()),
    )

    aggregate_metrics = build_aggregate_metrics(successful_rows, config.top_ks) if successful_rows else {}
    plots = build_result_plots(root_dirs, successful_rows, config.perf_type) if successful_rows else {}
    best_hyperparameters = {
        "selected_min_context_ply": int(best_stage_a["min_context_ply"]),
        "learning_rate": float(best_stage_b["learning_rate"]),
        "num_train_epochs": int(best_stage_b["num_train_epochs"]),
        "lora_rank": int(best_stage_b["lora_rank"]),
        "lora_alpha": int(best_stage_b["lora_alpha"]),
        "lora_dropout": float(best_stage_b["lora_dropout"]),
        "target_modules": str(best_stage_b["target_modules"]),
    }

    comparison_summary = {
        "run_id": run_name,
        "config": asdict(config),
        "baseline_validation": baseline_summary,
        "best_stage_a": best_stage_a,
        "best_stage_b": best_stage_b,
        "best_hyperparameters": best_hyperparameters,
        "aggregate_metrics": aggregate_metrics,
        "successful_players": successful_rows,
        "failed_players": failed_rows,
        "plots": plots,
    }
    write_json(root_dirs["root"] / "comparison_summary.json", comparison_summary)
    write_json(root_dirs["root"] / "aggregate_metrics.json", aggregate_metrics)
    write_json(root_dirs["root"] / "best_hyperparameters.json", best_hyperparameters)
    write_json(
        root_dirs["root"] / "final_summary.json",
        {
            "run_id": run_name,
            "best_hyperparameters": best_hyperparameters,
            "aggregate_metrics": aggregate_metrics,
            "successful_player_count": len(successful_rows),
            "failed_player_count": len(failed_rows),
            "plots": plots,
        },
    )

    if successful_rows:
        csv_path = root_dirs["root"] / "comparison_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(successful_rows[0].keys()))
            writer.writeheader()
            writer.writerows(successful_rows)

    print("Best stage A:", best_stage_a["candidate_name"])
    print("Best stage B:", best_stage_b["candidate_name"])
    print("Successful players:", len(successful_rows))
    print("Failed players:", len(failed_rows))
    print("Saved results to:", root_dirs["root"])

    return {
        "run_id": run_name,
        "results_dir": str(root_dirs["root"]),
        "best_stage_a": best_stage_a,
        "best_stage_b": best_stage_b,
        "best_hyperparameters": best_hyperparameters,
        "aggregate_metrics": aggregate_metrics,
        "comparison_summary": comparison_summary,
        "successful_rows": successful_rows,
        "failed_rows": failed_rows,
        "stage_a_rows": stage_a_rows,
        "stage_b_rows": stage_b_rows,
    }
