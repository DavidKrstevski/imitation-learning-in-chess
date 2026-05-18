from typing import List

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_service import choose_model_move, choose_player_model_move, load_model
from backend.training_service import (
    get_training_job,
    list_trained_players,
    list_training_jobs,
    start_training_job,
)

app = FastAPI(title="Chess Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MoveRequest(BaseModel):
    moves: List[str]
    player_id: str | None = None


class MoveResponse(BaseModel):
    model_move: str
    game_over: bool
    result: str | None = None


class TrainPlayerRequest(BaseModel):
    username: str


tokenizer = None
model = None
device = None


@app.on_event("startup")
def startup_event() -> None:
    global tokenizer, model, device
    tokenizer, model, device = load_model()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": str(device)}


@app.get("/api/players")
def players() -> dict:
    return {"players": list_trained_players()}


@app.get("/api/training-jobs")
def training_jobs() -> dict:
    return {"jobs": list_training_jobs()}


@app.post("/api/train-player")
def train_player(payload: TrainPlayerRequest) -> dict:
    try:
        return start_training_job(payload.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/train-player/{job_id}")
def train_player_status(job_id: str) -> dict:
    try:
        return get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown training job") from exc


@app.post("/api/model-move", response_model=MoveResponse)
def model_move(payload: MoveRequest) -> MoveResponse:
    board = chess.Board()

    for uci in payload.moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid UCI move: {uci}") from exc

        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail=f"Illegal move in history: {uci}")

        board.push(move)

    if board.is_game_over():
        return MoveResponse(model_move="", game_over=True, result=board.result())

    if payload.player_id:
        try:
            chosen = choose_player_model_move(board, payload.player_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        chosen = choose_model_move(board, tokenizer, model, device)
    return MoveResponse(model_move=chosen, game_over=False)
