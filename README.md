# chessathon

A chess engine built for [AI Chessathon](https://aichessathon.com), with a
from-scratch bitboard search, a team-trained HalfKP neural evaluator, and local
opening and endgame data. Python and Numba provide the complete playing runtime.

**Final reported competition rating: 1955 Elo. Project complete, 11 September 2026.**
The rating was reported by the team after submission; local validation results
are recorded separately below.

## Engine

- **Search:** iterative deepening, alpha-beta/PVS, transposition tables,
  aspiration windows, null-move pruning, late-move reductions and time management.
- **Evaluation:** a width-128 HalfKP network trained from random initialization,
  with incremental accumulators, integer inference and exact evaluation caching.
  The normal blend is 75% neural and 25% classical; positions with seven or fewer
  pieces and bare-king endings use the classical route.
- **Fast inference:** contiguous transposed dense weights, zero-activation
  skipping and reusable integer scratch storage preserve the original outputs.
- **Openings and endgames:** a 294,743-entry Polyglot opening book, local Syzygy
  tables and a verified 9,103-position rook-and-pawn-versus-rook winning policy.
  Coverage is limited to the supplied data.
- **Draw handling:** game-history tracking, repetition and fifty-move checks,
  plus history-aware selection of endgame moves.

The submitted engine runs on one CPU core without network access or a GPU. Its
neural weights were trained by the team; offline teacher engines were used to
label training positions and are excluded from the playing runtime. See the
[model card](docs/HALFKP_MODEL_CARD.md) and
[parent training record](docs/HALFKP_PARENT_MODEL_CARD_20260910.md) for provenance.

## Final validation

The final frozen submission was compared with the **immediately preceding
corrected upload (`c1b6dc4`)** over 24 colour-swapped opening pairs. Each game used
a fresh process, one CPU core and the unchanged competition referee at
120 seconds plus 0.5 seconds per move.

| Check | Recorded result |
|---|---|
| Direct match | **13 wins, 27 draws, 8 losses** in 48 games; **55.21%** score |
| Search throughput | **47.8% more nodes/second**, with identical fixed-depth moves, scores, nodes and search state |
| Win At Chess tactics | **277/300** at one second per position |
| Incremental evaluation | **105,643** position checks and **50,111** legal transitions; zero mismatches |
| Draw handling | **3,468** comparisons against the referee passed |
| Rook-ending conversions | **20/20** checkmates against exact DTZ defence |
| Runtime and source checks | All 48 PGNs replay legally; no flags, illegal moves, crashes or load failures; Ruff and strict mypy passed |

The match's paired statistical test was **inconclusive**. Its score corresponds
descriptively to about +36 local Elo; this is separate from the reported
competition rating and does not establish a 200-500 Elo gain. Throughput was
measured on an i7-10700K and is not a platform CPU measurement.

Of the 27 draws, 26 were repetitions and one was stalemate. An offline post-match
tablebase check confirmed that the stalemating move was the only move that saved
that game. Full methods, rejected experiments and results are in the
[final validation report](docs/TWO_HOUR_REFINEMENT_20260911.md), with
[raw evidence](docs/validation/refinement-twohour-20260911/) and a
[SHA-256 manifest](docs/validation/refinement-twohour-20260911/SHA256.json).

## Run locally

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```sh
git clone https://github.com/Cheekywnl/chessathon.git
cd chessathon
uv sync --locked
uv run python -m harness.play --white . --black baselines/greedy
```

Dependencies are pinned in `uv.lock`. The agent interface returns a UCI move for
the side to move:

```python
def get_move(fen: str, time_left_ms: int) -> str: ...
```

Models load and Numba kernels warm during import. The process retains game state
between moves and starts fresh for each game.

Useful development commands, on systems with Make and Bash:

```sh
make gate       # lint, strict typing and two short protocol games
make wac        # tactical regression suite
make endgame    # technical endgame regressions
make arena      # quick games against the development baseline
```

The bundled random, greedy and minimax baselines are development checks. Use
paired games against a frozen engine version to assess playing-strength changes.

## Submission package

The preserved final artifact is **`submission-final-locked.zip`**. Its source and
validation release is [`ab515ec`](https://github.com/Cheekywnl/chessathon/commit/ab515ecf11125773aa3ca4b369e035ebbbeaeae2);
the final README update leaves the playing code and assets unchanged.

| Package measurement | Bytes |
|---|---:|
| ZIP file | 41,332,226 |
| Outer unzipped contents | 45,455,743 |
| Fully expanded, including nested NPZ/ZIP/gzip | **49,042,751** |
| Margin below 50,000,000 | **957,249** |

The package contains 114 runtime files. Its SHA-256 is:

```text
2b25f386bcb7c4a795c43449cdc8c937f66f09c97763b5d56c181f4661b65719
```

To build a submission from a clean checkout:

```sh
uv run python -m harness.package --include syzygy --include book
uv run python -m tools.portable_submission --source-zip submission.zip --out submission-portable.zip --manifest submission-manifest.json
```

The second command verifies the recursively expanded size, checks the payload
and writes portable archive metadata. It requires a new output path.
`agent.py` sits at the archive root; the package includes the selected weights,
book and endgame assets. Upload the resulting submission archive: GitHub's
whole-repository download also contains development tools and evidence and is
not the competition package. The preserved ZIP above remains the reference for
the completed tests.

The official [agent contract](https://aichessathon.com/docs/agent-contract.md) and
[competition rules](https://aichessathon.com/docs/rules.md) define the runtime and
submission limits; the platform's validation log determines upload acceptance.

## Repository guide

| Path | Purpose |
|---|---|
| `agent.py` | Move selection, clock management, book and endgame integration |
| `chess_search.py`, `chess_movegen.py`, `chess_state.py` | Search, bitboards, legal moves and game state |
| `chess_eval.py`, `chess_halfkp_int.py`, `chess_nnue_halfkp.py` | Classical and neural evaluation |
| `chess_draw.py`, `chess_rook_endgame.py` | Draw safeguards and rook-ending policy |
| `weights/`, `book/`, `syzygy/` | Selected runtime assets |
| `harness/`, `baselines/` | Original competition protocol, referee and local opponents |
| `tools/` | Training, quantization, benchmarks, match analysis and package checks |
| `docs/` | Model provenance, experiment history and validation evidence |

The project builds on the AI Chessathon starter. Its protocol and referee remain
unchanged so local games use the competition interface and clock semantics.
