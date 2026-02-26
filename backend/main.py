from typing import List

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_service import load_model, choose_model_move

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


class MoveResponse(BaseModel):
    model_move: str
    game_over: bool
    result: str | None = None


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

    chosen = choose_model_move(board, tokenizer, model, device)
    return MoveResponse(model_move=chosen, game_over=False)
