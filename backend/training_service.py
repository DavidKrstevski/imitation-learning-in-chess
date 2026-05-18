from __future__ import annotations

import json
import shutil
import sys
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_MODELS_DIR = PROJECT_ROOT / "app_models"
APP_TRAINING_RUNS_DIR = APP_MODELS_DIR / "_training_runs"

if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from test_result_utils import ExperimentConfig, run_fixed_train_games_comparison, slugify_name  # noqa: E402


@dataclass
class TrainingJob:
    id: str
    username: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results_dir: str | None = None
    player_id: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


_jobs: dict[str, TrainingJob] = {}
_lock = threading.Lock()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _player_id(player_dir: Path) -> str:
    return player_dir.name


def _metric_summary(final_report: dict[str, Any]) -> dict[str, Any]:
    dataset = final_report.get("dataset_summary") or {}
    baseline = final_report.get("baseline_test_metrics") or {}
    finetuned = final_report.get("finetuned_test_metrics") or {}
    comparison = final_report.get("comparison") or {}
    profile = final_report.get("profile") or {}
    return {
        "elo": profile.get("elo"),
        "games": profile.get("games"),
        "train_games_used": dataset.get("train_games_used"),
        "test_games": dataset.get("test_games"),
        "test_examples": dataset.get("test_examples"),
        "baseline_top1_accuracy": baseline.get("top1_accuracy"),
        "finetuned_top1_accuracy": finetuned.get("top1_accuracy"),
        "finetuned_top3_accuracy": finetuned.get("top3_accuracy"),
        "finetuned_top5_accuracy": finetuned.get("top5_accuracy"),
        "delta_top1_accuracy": comparison.get("delta_top1_accuracy"),
        "finetuned_mean_rank": finetuned.get("mean_rank"),
    }


def list_trained_players() -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    if not APP_MODELS_DIR.exists():
        return players

    for final_report_path in APP_MODELS_DIR.glob("*/final_report.json"):
        player_dir = final_report_path.parent
        if player_dir.name.startswith("_"):
            continue
        best_model_dir = player_dir / "best_model"
        if not best_model_dir.exists():
            continue
        final_report = _read_json(final_report_path) or {}
        username = str(final_report.get("username") or player_dir.name)
        players.append(
            {
                "id": _player_id(player_dir),
                "username": username,
                "player_dir": str(player_dir),
                "model_dir": str(best_model_dir),
                "metrics": _metric_summary(final_report),
            }
        )

    players.sort(key=lambda row: row["username"].lower(), reverse=False)
    return players


def get_player_model_dir(player_id: str) -> Path:
    for player in list_trained_players():
        if player["id"] == player_id:
            return Path(player["model_dir"])
    raise KeyError(f"Unknown trained player id: {player_id}")


def _latest_status_for_username(username: str) -> dict[str, Any] | None:
    slug = slugify_name(username)
    status_files = sorted(
        [
            *APP_MODELS_DIR.glob(f"{slug}/status.json"),
            *APP_TRAINING_RUNS_DIR.glob(f"*/players/{slug}/status.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in status_files:
        payload = _read_json(path)
        if payload:
            return payload
    return None


def _refresh_job_from_disk(job: TrainingJob) -> None:
    status_payload = _latest_status_for_username(job.username)
    if not status_payload:
        return
    job.stage = str(status_payload.get("stage") or job.stage)
    progress = status_payload.get("progress")
    if isinstance(progress, (int, float)):
        job.progress = max(job.progress, float(progress))
    job.updated_at = str(status_payload.get("updated_at") or job.updated_at)


def get_training_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        _refresh_job_from_disk(job)
        return asdict(job)


def list_training_jobs() -> list[dict[str, Any]]:
    with _lock:
        for job in _jobs.values():
            _refresh_job_from_disk(job)
        return [asdict(job) for job in sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)]


def _single_player_config(username: str) -> ExperimentConfig:
    return ExperimentConfig(
        player_usernames=(username,),
        max_games=500,
        train_games_per_player=500,
        perf_type="classical",
        split_strategy="chronological",
        test_frac=0.1,
        val_frac_within_train=0.05,
        min_context_ply=0,
        learning_rate=2.5e-4,
        num_train_epochs=1,
        lora_rank=32,
        lora_alpha=64,
        target_modules="all-linear",
        require_train_games_per_player=False,
        results_root_name=str(APP_TRAINING_RUNS_DIR),
    )


def _publish_player_dir(training_player_dir: Path, username: str) -> Path:
    slug = slugify_name(username)
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = APP_MODELS_DIR / slug
    replacement_dir = APP_MODELS_DIR / f".{slug}.replacement"
    if replacement_dir.exists():
        shutil.rmtree(replacement_dir, ignore_errors=True)
    shutil.copytree(training_player_dir, replacement_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    replacement_dir.rename(target_dir)
    return target_dir


def _run_training_job(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        job.status = "running"
        job.stage = "starting"
        job.updated_at = datetime.now(timezone.utc).isoformat()

    try:
        username = _jobs[job_id].username
        result = run_fixed_train_games_comparison(_single_player_config(username))
        successful_rows = result.get("successful_rows") or []
        failed_rows = result.get("failed_rows") or []

        if not successful_rows:
            message = "Training finished without a successful player result."
            if failed_rows:
                message = str(failed_rows[0].get("error_message") or message)
            raise RuntimeError(message)

        result_dir = Path(str(result.get("results_dir") or ""))
        training_player_dir = result_dir / "players" / slugify_name(username)
        player_dir = _publish_player_dir(training_player_dir, username)
        final_report = _read_json(player_dir / "final_report.json") or {}
        player = None
        if final_report and (player_dir / "best_model").exists():
            player = {
                "id": _player_id(player_dir),
                "metrics": _metric_summary(final_report),
            }
        metrics = player["metrics"] if player else successful_rows[0]
        with _lock:
            job = _jobs[job_id]
            job.status = "completed"
            job.stage = "completed"
            job.progress = 1.0
            job.results_dir = str(player_dir)
            job.player_id = player["id"] if player else None
            job.metrics = metrics
            job.updated_at = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.stage = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.updated_at = datetime.now(timezone.utc).isoformat()
            job.metrics = {"traceback": traceback.format_exc()}


def start_training_job(username: str) -> dict[str, Any]:
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("username must not be empty")

    job = TrainingJob(id=uuid.uuid4().hex, username=clean_username)
    with _lock:
        _jobs[job.id] = job

    thread = threading.Thread(target=_run_training_job, args=(job.id,), daemon=True)
    thread.start()
    return asdict(job)
