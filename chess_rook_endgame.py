"""Partial exact five-piece rook-and-pawn winning policy, derived from Syzygy.

Keys encode every square without hashing. Positions outside the recorded graph
return None. Colour and file symmetry share entries; DTZ bounds respect the
remaining fifty-move window. Real-game repetition claims are checked by agent.py.
"""

import sys
from pathlib import Path

import chess
import numpy as np


def _load() -> dict[int, tuple[int, int]]:
    try:
        with np.load(Path(__file__).resolve().parent / "syzygy/rook_wins.npz",
                     allow_pickle=False) as data:
            if int(data["format_version"]) != 1:
                raise ValueError("unsupported rook policy format")
            return {int(key): (int(move), int(dtz)) for key, move, dtz in
                    zip(data["keys"], data["moves"], data["dtz"], strict=True)}
    except Exception as exc:
        print(f"rook policy unavailable, continuing with search: {exc}", file=sys.stderr)
        return {}


_WINS = _load()


def winning_move(board: chess.Board) -> chess.Move | None:
    """Return a legal recorded win, with safe DTZ, only for exact KRP versus KR."""
    if board.occupied.bit_count() != 5 or board.castling_rights:
        return None
    us = board.turn
    pawn = board.pieces_mask(chess.PAWN, us)
    our_rook = board.pieces_mask(chess.ROOK, us)
    their_rook = board.pieces_mask(chess.ROOK, not us)
    our_king = board.king(us)
    their_king = board.king(not us)
    if (pawn.bit_count() != 1 or our_rook.bit_count() != 1
            or their_rook.bit_count() != 1 or our_king is None or their_king is None):
        return None
    flip = 0 if us == chess.WHITE else 56
    pawn_square = pawn.bit_length() - 1
    if chess.square_file(pawn_square) >= 4:
        flip ^= 7
    squares = (our_king, our_rook.bit_length() - 1, pawn_square,
               their_king, their_rook.bit_length() - 1)
    key = sum((square ^ flip) << (6 * index) for index, square in enumerate(squares))
    entry = _WINS.get(key)
    if entry is None:
        return None
    packed, dtz = entry
    if board.halfmove_clock + dtz >= 100:
        return None
    move = chess.Move((packed & 63) ^ flip, ((packed >> 6) & 63) ^ flip,
                      promotion=(packed >> 12) or None)
    return move if move in board.legal_moves else None
