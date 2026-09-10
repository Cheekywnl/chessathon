"""Fingerprint tested builds and audit completed real-protocol game records."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import chess.pgn

from harness.referee import FAILED_TERMINATIONS, RESULT_HEADERS
from tools.halfkp_data import file_hash

ERROR_MARKERS = ("traceback", "load failed", "probe failed", "get_move crashed",
                 "exception in thread", "falling back to first legal")


def build_manifest(root: Path) -> dict[str, object]:
    files = sorted(root.glob("*.py"))
    for directory in ("book", "syzygy", "weights"):
        files.extend(sorted(path for path in (root / directory).rglob("*") if path.is_file()))
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"path": str(root.resolve()), "runtime_sha256": digest, "files": hashes}


def known_terminal_opponent_ponder(row: dict[str, object], log: str) -> bool:
    """Recognize the reproduced baseline diagnostic after its last move before mate.

    This never permits a candidate error, failed outcome or earlier error. The
    opponent has already chosen its final legal move and the game then ends in
    mate, so a subsequent worker diagnostic cannot change that completed outcome.
    Tracebacks can be truncated when the referee kills the finished process;
    retain the entire diagnostic without claiming every truncated cause is proved.
    """
    marker = "Exception in thread"
    if log.count(marker) != 1 or row["termination"] != "checkmate":
        return False
    if (row["result"] == "white") != row["agent_white"]:
        return False
    before, tail = log.split(marker)
    if any(word in before.lower() for word in ERROR_MARKERS):
        return False
    if re.match(r" Thread-\d+ \(_ponder\):", tail) is None or "move=" in tail:
        return False
    game = chess.pgn.read_game(io.StringIO(str(row["pgn"])))
    if game is None:
        return False
    moves = list(game.mainline_moves())
    logged_moves = re.findall(r"move=(\S+)", before)
    return (len(moves) >= 2 and bool(logged_moves) and logged_moves[-1] == moves[-2].uci()
            and game.end().board().is_checkmate())


def audit_runtime_logs(row: dict[str, object]) -> list[str]:
    diagnostics = []
    for side in ("white_log", "black_log"):
        log = str(row[side])
        if not any(marker in log.lower() for marker in ERROR_MARKERS):
            continue
        is_candidate = (side == "white_log") == row["agent_white"]
        if not is_candidate and known_terminal_opponent_ponder(row, log):
            diagnostics.append("baseline ponder exception after its final move before checkmate")
        else:
            raise ValueError(f"unexplained runtime error in {side}: {log[-2500:]}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for path in args.logs
               for line in path.read_text(encoding="utf8").splitlines() if line]
    fingerprints = set()
    fingerprinted_logs = 0
    for path in args.logs:
        sidecar = path.with_suffix(".builds.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf8"))
            builds = metadata.get("builds", metadata)
            fingerprints.add((builds["agent"]["runtime_sha256"],
                              builds["opponent"]["runtime_sha256"]))
            fingerprinted_logs += 1
    assert len(fingerprints) <= 1, "do not aggregate different runtime builds"
    keys = set()
    scores = []
    terminations: Counter[str] = Counter()
    paired: dict[str, list[float]] = {}
    diagnostics: list[dict[str, object]] = []
    for row in records:
        key = (row["opening"], row["agent_white"])
        assert key not in keys, "repeated opening/colour would overstate sample diversity"
        keys.add(key)
        assert row["termination"] not in FAILED_TERMINATIONS and row["result"] != "void"
        notes = audit_runtime_logs(row)
        if notes:
            diagnostics.append({"opening": row["opening"], "agent_white": row["agent_white"],
                                "notes": notes})
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        assert game is not None and not game.errors
        assert game.headers["Result"] == RESULT_HEADERS[row["result"]]
        board = game.end().board()
        outcome = board.outcome(claim_draw=True)
        if row["termination"] != "ply_cap":
            assert outcome is not None and outcome.termination.name.lower() == row["termination"]
            assert outcome.result() == game.headers["Result"]
        else:
            assert board.ply() == row.get("ply_cap_total", 600)
        score = (0.5 if row["result"] == "draw" else
                 float((row["result"] == "white") == row["agent_white"]))
        scores.append(score)
        paired.setdefault(row["opening"], []).append(score)
        terminations[row["termination"]] += 1
    assert scores and all(len(pair) == 2 for pair in paired.values()), "unfinished colour pairs"
    clocks = sorted({(row["base_ms"], row["increment_ms"]) for row in records})
    assert len(clocks) == 1, "do not aggregate different clocks"
    result = {"games": len(scores), "openings": len(paired), "clock_ms": clocks[0],
              "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
              "score": sum(scores) / len(scores), "terminations": dict(terminations),
              "pair_score_counts": dict(Counter(str(sum(pair)) for pair in paired.values())),
              "log_sha256": {str(path): file_hash(path) for path in args.logs},
              "fingerprinted_logs": fingerprinted_logs,
              "runtime_fingerprints": [list(pair) for pair in fingerprints],
              "maximum_recorded_peak_rss_bytes": max(
                  (row.get(key) or 0 for row in records
                   for key in ("white_peak_rss", "black_peak_rss")), default=0),
              "maximum_recorded_init_seconds": max(
                  (row.get(key) or 0 for row in records
                   for key in ("white_init_seconds", "black_init_seconds")), default=0),
              "legal_pgn_replay": True, "candidate_failure_or_fallback_markers": 0,
              "baseline_diagnostics": diagnostics,
              "platform_elo_established": False}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
