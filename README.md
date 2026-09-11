Current candidate: [11 September final refinement and validation](docs/TWO_HOUR_REFINEMENT_20260911.md).

This fork contains the team's own compiled alpha-beta/PVS engine and a HalfKP
network trained from random initialization. The current candidate adds exact
sparse neural inference, reusable scratch storage and a verified 9,103-position
rook-and-pawn winning policy. Its actual submission ZIP expands recursively to
49,042,751 bytes. The final 48-game comparison against the previous corrected
upload finished 13W/27D/8L (55.21% score) at 120s+0.5s.
Measured search speed is separate from playing-strength evidence.

Upload the packaged submission, not GitHub's repository ZIP. The repository also
contains tools, baselines and validation records that are not submission files.

Selected network: [training and model provenance](docs/HALFKP_MODEL_CARD.md).

# Agent and local harness

The project builds on the [AI Chessathon](https://aichessathon.com) starter's
unchanged protocol, referee and packaging harness. Baselines and the harness are
development tools; the submitted runtime uses the root Python modules and the
selected assets in `weights/`, `book/` and `syzygy/`.

```
git clone --branch final-release https://github.com/Cheekywnl/chess-codex-isolated-20260910.git
cd chess-codex-isolated-20260910
make setup
make play
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, `make zip` and drop `submission.zip` on your dashboard.

## Writing an agent

`agent.py` is the submission entrypoint. Its public interface is:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The engine returns legal UCI moves through this interface. The separate random
baseline is useful for protocol checks; beating it does not establish strength.

```
make play                                          # one game, real time control
make arena                                         # 20 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything your agent writes to stdout or stderr shows up under the result, so `print` debugging
works. The platform discards it during rated games and shows it in your validation log.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py                        submission entrypoint: time management, book/tablebase
                                 lookup, the repetition backstop, and a crash safety net
                                 around move selection
chess_search.py                 negamax/PVS search: TT, null-move, LMR, futility,
                                 aspiration windows
chess_eval.py                   tapered material + PST + structure eval, tunable PARAMS
chess_movegen.py                from-scratch numba-jitted bitboard move generator
chess_state.py                  raw bitboard game state, Zobrist hashing, SEE
syzygy/                         Syzygy endgame tablebases (3-4 piece, plus a few common
                                 5-piece endings): provably perfect play once few enough
                                 pieces remain, shipped data (see AGENTS.md)
book/                           a Polyglot opening book, shipped data, an early-blunder
                                 safety net rather than a strength source (see AGENTS.md)
baselines/                      random, greedy, minimax, numba; each a directory with
                                 its own agent.py
harness/runner.py               the process the platform runs your agent in
harness/referee.py              the clock, legality, draw and adjudication rules
harness/rules.py                the event constants the harness enforces
harness/sandbox.py              the one process, spoken to as the platform speaks to it
harness/play.py                 one game between two agent directories
harness/arena.py                many games, with a score
harness/package.py              builds submission.zip; `make zip` passes --include for
                                 syzygy/ and book/ so they end up in it too
tools/fast_arena.py             in-process A/B arena for eval/param changes -- no
                                 subprocess spawn or JIT re-warm per game
tools/wac_test.py, wac.epd      the Win At Chess tactical regression suite
tools/endgame_regression.py     known-hard technical endgames played through the real
                                 get_move, not a bypassed direct-search shortcut
tools/generate_training_data.py,
tools/tune.py                   Texel tuning pipeline (see tune.py's own docstring)
docs/IDEAS.md                   where the strength actually comes from
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
