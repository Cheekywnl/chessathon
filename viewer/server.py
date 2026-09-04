"""Local-only game viewer. Never part of the submission -- package.py only zips *.py files
at the project root, this directory is never touched by `make zip`.

Runs our agent in-process (so we can capture its per-move stderr stats directly) against a
chosen opponent through the real subprocess sandbox (so the opponent's behaviour is exactly
what the platform would see), and serves live board state + search stats to a browser.

Run with: uv run python viewer/server.py   (from the chessathon project root)
Then open http://127.0.0.1:8765
"""

import contextlib
import io
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.svg  # noqa: E402

import agent as our_agent  # noqa: E402
import chess_search as cs  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402

PORT = 8765
OUR_TT_SIZE_POWER = 21

OPPONENTS = {
    "random": ROOT / "baselines" / "random",
    "greedy": ROOT / "baselines" / "greedy",
    "minimax": ROOT / "baselines" / "minimax",
    "stockfish": ROOT / "baselines" / "stockfish",
}

STATS_RE = re.compile(
    r"move=(\S+) depth=(\d+) score=(-?\d+) nodes=(\d+) ms=(\d+) time_left_ms=(\d+)"
)

_lock = threading.Lock()
_state: dict[str, object] = {
    "status": "idle",  # idle | running | finished
    "fen": chess.STARTING_FEN,
    "moves": [],  # {ply, san, uci, is_us, depth, score, nodes, ms}
    "result": None,
    "termination": None,
    "opponent": None,
    "we_play_white": True,
    "error": None,
}
# Completed games, newest first: {opponent, we_play_white, result, termination, plies}
_history: list[dict] = []


def _reset_our_agent_state() -> None:
    """Simulate what a fresh process gives the real submission for free: a game's worth of
    state (TT, position history, any live ponder thread) never survives to the next game."""
    our_agent._stop_pondering()
    our_agent._TT = cs.TranspositionTable(size_power=OUR_TT_SIZE_POWER)
    our_agent._GAME_HISTORY.clear()


def _our_move(fen: str, time_left_ms: int) -> tuple[str, dict[str, int | None]]:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        uci = our_agent.get_move(fen, time_left_ms)
    stats: dict[str, int | None] = {"depth": None, "score": None, "nodes": None, "ms": None}
    match = None
    for line in buf.getvalue().splitlines():
        found = STATS_RE.search(line)
        if found:
            match = found
    if match:
        stats = {
            "depth": int(match.group(2)),
            "score": int(match.group(3)),
            "nodes": int(match.group(4)),
            "ms": int(match.group(5)),
        }
    return uci, stats


def run_game(opponent_key: str, we_play_white: bool, base_ms: int, increment_ms: int) -> None:
    _reset_our_agent_state()
    board = chess.Board()
    opponent = local(OPPONENTS[opponent_key])

    with _lock:
        _state.update(
            status="running",
            fen=board.fen(),
            moves=[],
            result=None,
            termination=None,
            opponent=opponent_key,
            we_play_white=we_play_white,
            error=None,
        )

    def finish(result: str, termination: str) -> None:
        with _lock:
            _state.update(status="finished", result=result, termination=termination)
            _history.insert(
                0,
                {
                    "opponent": opponent_key,
                    "we_play_white": we_play_white,
                    "result": result,
                    "termination": termination,
                    "plies": len(_state["moves"]),
                },
            )

    try:
        opponent.start(60.0)
    except AgentFailure as e:
        finish("opponent_failed_to_start", e.reason)
        return

    clocks = {True: float(base_ms), False: float(base_ms)}  # keyed by is_our_turn

    try:
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.winner is None:
                    finish("draw", outcome.termination.name.lower())
                else:
                    winner_is_us = outcome.winner == we_play_white
                    finish("us" if winner_is_us else "opponent", outcome.termination.name.lower())
                return
            if len(board.move_stack) >= PLY_CAP:
                finish("adjudication", "adjudication")
                return

            is_our_turn = (board.turn == chess.WHITE) == we_play_white
            time_left = clocks[is_our_turn]
            started_at = time.monotonic()

            if is_our_turn:
                uci, stats = _our_move(board.fen(), int(time_left))
            else:
                try:
                    uci = opponent.move(board.fen(), int(time_left))
                except AgentFailure as e:
                    finish("us", e.reason)
                    return
                stats = {}

            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            clocks[is_our_turn] -= elapsed_ms
            if clocks[is_our_turn] < 0:
                finish("opponent" if is_our_turn else "us", "flag")
                return

            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError:
                finish("opponent" if is_our_turn else "us", "illegal")
                return
            if move not in board.legal_moves:
                finish("opponent" if is_our_turn else "us", "illegal")
                return

            san = board.san(move)
            board.push(move)
            clocks[is_our_turn] += increment_ms

            with _lock:
                _state["fen"] = board.fen()
                _state["moves"].append(
                    {
                        "ply": len(_state["moves"]) + 1,
                        "san": san,
                        "uci": uci,
                        "is_us": is_our_turn,
                        **stats,
                    }
                )
    finally:
        opponent.stop()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(ROOT / "viewer" / "index.html", "text/html")
        elif path == "/api/state":
            with _lock:
                self._send_json({**_state, "history": _history[:20]})
        elif path == "/api/board.svg":
            with _lock:
                fen = _state["fen"]
                moves = _state["moves"]
            board = chess.Board(fen)
            lastmove = chess.Move.from_uci(moves[-1]["uci"]) if moves else None
            svg = chess.svg.board(board, lastmove=lastmove, size=420).encode()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(svg)))
            self.end_headers()
            self.wfile.write(svg)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/game/start":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length)) if length else {}
        opponent = payload.get("opponent", "greedy")
        if opponent not in OPPONENTS:
            self._send_json({"error": f"unknown opponent {opponent!r}"}, status=400)
            return
        with _lock:
            if _state["status"] == "running":
                self._send_json({"error": "a game is already running"}, status=409)
                return
        threading.Thread(
            target=run_game,
            args=(
                opponent,
                bool(payload.get("we_play_white", True)),
                int(payload.get("base_ms", 15000)),
                int(payload.get("increment_ms", 200)),
            ),
            daemon=True,
        ).start()
        self._send_json({"ok": True})

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"viewer running at http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
