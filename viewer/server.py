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
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
import chess.svg  # noqa: E402

import agent as our_agent  # noqa: E402
import chess_search as cs  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402

PORT = 8765
OUR_TT_SIZE_POWER = 21
REAL_GAMES_DIR = ROOT / "games" / "real"
OUR_NAME = "Cheeky"
LOGS_DIR = ROOT / "data" / "logs"

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

_real_games_cache: list[dict] | None = None


def _round_number(path: Path) -> int:
    match = re.search(r"round-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def load_real_games() -> list[dict]:
    """Parse every PGN in games/real/ once and cache it -- these are real rated games
    downloaded from the dashboard, read-only reference data for the viewer."""
    global _real_games_cache
    if _real_games_cache is not None:
        return _real_games_cache

    games = []
    if REAL_GAMES_DIR.is_dir():
        for path in sorted(REAL_GAMES_DIR.glob("*.pgn"), key=_round_number):
            with open(path) as f:
                game = chess.pgn.read_game(f)
            if game is None:
                continue
            headers = game.headers
            white = headers.get("White", "?")
            black = headers.get("Black", "?")
            is_reference = OUR_NAME not in (white, black)
            we_play_white = white == OUR_NAME
            result = headers.get("Result", "*")
            if is_reference:
                # Neither side is us -- a reference game (e.g. a top-rated player's game
                # against someone else), not one of ours. Keep both names, don't score it.
                opponent = f"{white} vs {black}"
                our_result = "reference"
            else:
                opponent = black if we_play_white else white
                if result == "1/2-1/2":
                    our_result = "draw"
                elif (result == "1-0") == we_play_white:
                    our_result = "win"
                else:
                    our_result = "loss"

            board = game.board()
            moves = []
            for node in game.mainline():
                move = node.move
                san = board.san(move)
                clk = None
                if "%clk" in node.comment:
                    clk = node.comment.split("%clk")[1].split("]")[0].strip()
                board.push(move)
                is_us = not is_reference and (board.turn != chess.WHITE) == we_play_white
                moves.append(
                    {"san": san, "uci": move.uci(), "is_us": is_us, "clock": clk}
                )

            games.append(
                {
                    "round": _round_number(path),
                    "file": path.name,
                    "opponent": opponent,
                    "we_play_white": we_play_white,
                    "is_reference": is_reference,
                    "result": our_result,
                    "termination": headers.get("Termination", "?"),
                    "date": headers.get("Date", "?"),
                    "plies": len(moves),
                    "start_fen": headers.get("FEN", chess.STARTING_FEN),
                    "moves": moves,
                }
            )
    _real_games_cache = list(reversed(games))
    return _real_games_cache


def real_games_summary() -> dict:
    games = [g for g in load_real_games() if not g["is_reference"]]
    wins = sum(1 for g in games if g["result"] == "win")
    draws = sum(1 for g in games if g["result"] == "draw")
    losses = sum(1 for g in games if g["result"] == "loss")
    terminations: dict[str, int] = {}
    for g in games:
        terminations[g["termination"]] = terminations.get(g["termination"], 0) + 1
    score = (wins + draws * 0.5) / len(games) * 100 if games else 0.0
    return {
        "total": len(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": round(score, 1),
        "terminations": terminations,
    }


def get_changelog(limit: int = 30) -> list[dict]:
    try:
        fmt = "%H%x1f%ad%x1f%s%x1f%b%x1e"
        out = subprocess.run(
            ["git", "log", f"-{limit}", f"--pretty=format:{fmt}", "--date=short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    entries = []
    for record in out.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x1f")
        if len(parts) < 3:
            continue
        commit_hash, date, subject = parts[0], parts[1], parts[2]
        body = parts[3].strip() if len(parts) > 3 else ""
        entries.append({"hash": commit_hash[:7], "date": date, "subject": subject, "body": body})
    return entries


def get_head_info() -> dict:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%H%x1f%s"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        commit_hash, subject = out.split("\x1f", 1)
        return {"hash": commit_hash[:7], "subject": subject, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"hash": "?", "subject": "?", "dirty": False}


ACTIVITY_LOG_PATH = ROOT / "viewer" / "activity_log.jsonl"


def get_activity_log(limit: int = 100) -> list[dict]:
    if not ACTIVITY_LOG_PATH.exists():
        return []
    entries = []
    with open(ACTIVITY_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(entries[-limit:]))


# ---- background task progress (training runs, SPRT arenas, data downloads) ----
# Self-describing: reads whatever's in data/logs/*.out.log and infers everything (epoch counts,
# SPRT bounds, download limits) from the log content itself, so a new run started later tonight
# shows up automatically -- no registry to edit here each time a new job gets launched.

_EPOCH_RE = re.compile(r"epoch (\d+)/(\d+): train_loss=([\d.]+) val_loss=([\d.]+)")
_SPRT_GAME_RE = re.compile(r"^game (\d+) .*?score ([\d.]+)%, llr=(-?[\d.]+)")
_SPRT_BOUNDS_RE = re.compile(r"bounds=\[(-?[\d.]+), (-?[\d.]+)\]")
_SPRT_VERDICT_RE = re.compile(r"(H0 ACCEPTED|H1 ACCEPTED|INCONCLUSIVE)")
_DOWNLOAD_RE = re.compile(r"written (\d+)/(\d+)")


def _tail_lines(path: Path, max_bytes: int = 131072) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode(errors="ignore").splitlines()


def _humanize_log_name(name: str) -> str:
    stem = name
    for suffix in (".out.log", ".log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ")


def _has_error(log_path: Path) -> bool:
    err_path = log_path.with_suffix("").with_suffix(".err.log")
    lines = _tail_lines(err_path, max_bytes=8192)
    return any("Traceback" in line or "Error" in line for line in lines)


def _parse_training(path: Path) -> dict[str, object]:
    epoch = total_epochs = 0
    train_loss = val_loss = None
    device = None
    saved = False
    for line in _tail_lines(path):
        match = _EPOCH_RE.search(line)
        if match:
            epoch, total_epochs = int(match.group(1)), int(match.group(2))
            train_loss, val_loss = float(match.group(3)), float(match.group(4))
        if line.startswith("device:"):
            device = line.split(":", 1)[1].strip()
        if "saved weights" in line:
            saved = True
    status = "done" if saved else ("running" if (epoch or device) else "starting")
    pct = (epoch / total_epochs * 100) if total_epochs else 0.0
    return {
        "status": status, "progress_pct": round(pct, 1),
        "epoch": epoch, "total_epochs": total_epochs,
        "train_loss": train_loss, "val_loss": val_loss, "device": device,
    }


def _parse_sprt(path: Path) -> dict[str, object]:
    bounds: tuple[float, float] | None = None
    last_game = last_score = last_llr = None
    verdict = None
    for line in _tail_lines(path):
        match = _SPRT_BOUNDS_RE.search(line)
        if match:
            bounds = (float(match.group(1)), float(match.group(2)))
        match = _SPRT_GAME_RE.search(line)
        if match:
            last_game = int(match.group(1))
            last_score = float(match.group(2))
            last_llr = float(match.group(3))
        match = _SPRT_VERDICT_RE.search(line)
        if match:
            verdict = match.group(1)
    status = "done" if verdict else ("running" if last_game else "starting")
    pct = 100.0 if verdict else 0.0
    if not verdict and bounds and last_llr is not None:
        lower, upper = bounds
        pct = max(0.0, min(100.0, (last_llr - lower) / (upper - lower) * 100))
    return {
        "status": status, "progress_pct": round(pct, 1),
        "games": last_game, "score_pct": last_score, "llr": last_llr,
        "bounds": list(bounds) if bounds else None, "verdict": verdict,
    }


def _parse_download(path: Path) -> dict[str, object]:
    written = limit = 0
    done = False
    for line in _tail_lines(path):
        match = _DOWNLOAD_RE.search(line)
        if match:
            written, limit = int(match.group(1)), int(match.group(2))
        if "reached limit" in line or "stream ended" in line:
            done = True
    status = "done" if done else ("running" if written else "starting")
    pct = min(100.0, written / limit * 100) if limit else 0.0
    return {"status": status, "progress_pct": round(pct, 1), "written": written, "limit": limit}


def get_training_tasks() -> list[dict[str, object]]:
    if not LOGS_DIR.is_dir():
        return []
    results = []
    for path in sorted(LOGS_DIR.glob("*.out.log")):
        name = path.name
        if name.startswith("train_"):
            kind, info = "training", _parse_training(path)
        elif name.startswith("sprt_"):
            kind, info = "sprt", _parse_sprt(path)
        elif name.startswith("lichess_prep"):
            kind, info = "download", _parse_download(path)
        else:
            continue
        results.append({
            "id": path.stem, "label": _humanize_log_name(name), "kind": kind,
            "has_error": _has_error(path), **info,
        })
    return results


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
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

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
            self._send_svg(chess.svg.board(board, lastmove=lastmove, size=420))
        elif path == "/api/real_games":
            games = load_real_games()
            summary_only = [{k: v for k, v in g.items() if k != "moves"} for g in games]
            self._send_json(summary_only)
        elif path == "/api/analytics":
            self._send_json(
                {
                    "real_games": real_games_summary(),
                    "changelog_count": len(get_changelog(limit=200)),
                    "head": get_head_info(),
                }
            )
        elif path == "/api/changelog":
            self._send_json(get_changelog())
        elif path == "/api/activity":
            self._send_json(get_activity_log())
        elif path == "/api/training":
            self._send_json(get_training_tasks())
        elif path.startswith("/api/real_games/") and path.endswith("/board.svg"):
            round_no = int(path.split("/")[3])
            ply = int(query.get("ply", ["0"])[0])
            game = next((g for g in load_real_games() if g["round"] == round_no), None)
            if game is None:
                self.send_response(404)
                self.end_headers()
                return
            board = chess.Board(game["start_fen"])
            lastmove = None
            for m in game["moves"][:ply]:
                lastmove = chess.Move.from_uci(m["uci"])
                board.push(lastmove)
            self._send_svg(
                chess.svg.board(
                    board, lastmove=lastmove, flipped=not game["we_play_white"], size=420
                )
            )
        elif path.startswith("/api/real_games/"):
            round_no = int(path.rsplit("/", 1)[-1])
            game = next((g for g in load_real_games() if g["round"] == round_no), None)
            if game is None:
                self.send_response(404)
                self.end_headers()
                return
            self._send_json(game)
        else:
            self.send_response(404)
            self.end_headers()

    def _send_svg(self, svg: str) -> None:
        body = svg.encode()
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
