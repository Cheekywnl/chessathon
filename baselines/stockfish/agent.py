"""Local-only benchmarking opponent. Never ships: harness/package.py only zips root-level
files, this directory is never touched by `make zip`. Wraps the system Stockfish binary at a
calibrated Elo via UCI_LimitStrength, purely so local strength testing means something."""

import os

import chess
import chess.engine

_ELO = int(os.environ.get("SF_ELO", "1500"))
_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": _ELO, "Threads": 1, "Hash": 64})


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    moves_to_go = max(15, 45 - board.fullmove_number)
    think_s = max(0.05, min(time_left_ms / moves_to_go / 1000.0, 3.0))
    result = _engine.play(board, chess.engine.Limit(time=think_s))
    assert result.move is not None
    return result.move.uci()
