"""Reproduce the baseline's terminal predicted-reply ponder error from a saved match."""

from __future__ import annotations

import argparse
import io
import json
import threading
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn

import agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = next(json.loads(line) for line in args.log.read_text(encoding="utf8").splitlines()
               if "Exception in thread" in line)
    game = chess.pgn.read_game(io.StringIO(row["pgn"]))
    assert game is not None
    moves = list(game.mainline_moves())
    board = game.board()
    for move in moves[:-2]:
        board.push(move)
    before_fen = board.fen()
    terminal = board.copy()
    terminal.push(moves[-2])
    terminal.push(moves[-1])
    assert terminal.is_checkmate()
    try:
        agent._ponder(terminal, threading.Event())
    except IndexError as exc:
        reproduced = str(exc)
    else:
        raise AssertionError("expected the old unguarded terminal ponder to raise IndexError")
    with patch.object(agent, "_predict_reply", return_value=moves[-1]), \
            patch.object(threading, "Thread") as spawn:
        agent._start_pondering(board, moves[-2])
        spawn.assert_not_called()
    # A legal, non-terminal prediction still starts the original worker.
    opening = chess.Board()
    with patch.object(agent, "_predict_reply", return_value=chess.Move.from_uci("e7e5")), \
            patch.object(threading, "Thread") as spawn:
        agent._start_pondering(opening, chess.Move.from_uci("e2e4"))
        spawn.assert_called_once()
    result = {"before_last_two_plies": before_fen,
              "last_two_moves": [move.uci() for move in moves[-2:]],
              "terminal_fen": terminal.fen(), "old_terminal_ponder_error": reproduced,
              "terminal_prediction_does_not_spawn": True,
              "nonterminal_prediction_still_spawns": True}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
