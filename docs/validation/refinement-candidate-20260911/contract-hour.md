# Agent contract

The participant-facing interface, rendered at `/docs` on the site together with the rules. It does not change once the qualifier starts.

## What you submit

A zip whose root holds

- `agent.py`, exposing `get_move(fen: str, time_left_ms: int) -> str` and returning a UCI move such as `e2e4` or `e7e8q`
- model weights, opening books and any other files, at most 50 MB unzipped in total

The root means the top of the archive, not a folder inside it. Most zip tools wrap the folder you selected, so check before uploading. A zip without `agent.py` at its root is rejected.

## The environment

- Python 3.12 with torch, numpy, python-chess, onnxruntime and numba preinstalled at the fixed versions the docs page lists. Nothing else installs and a `requirements.txt` in the zip is ignored.
- torch is the CPU build. The full standard library is there.
- An import outside that set crashes your agent in its smoke games.
- Ask hello@aichessathon.com for a package the stack lacks. Any addition is announced to every team.
- Compiled speed comes from the stack, not from your zip. numba JIT compiles your Python in process, and each jitted function pays that compile cost the first time it runs.
- Cython does not work here. A compiled extension is a native binary, which is rejected, and the image carries no compiler to build one at runtime.

## What may be in the zip

- Native binaries in the zip are rejected. Model weights are not binaries, so `.onnx`, `.safetensors` and `.pt` are fine.
- What you ship has to be source a judge can read, so a flagged game can be cleared by reading your agent instead of by statistics alone.
- Any network you ship is one you trained yourself. Starting from a published chess network is not allowed, so fine-tuning or re-exporting one counts as shipping it.
- File modes inside the zip are ignored and every file is readable by your process.
- Your zip is first on `sys.path`, so a file named after a module you import, like `chess.py` or `types.py`, shadows the real one.

## How your process runs

- One process per agent per game, started fresh for every game.
- Load your model at import. The 90 s init budget covers importing your agent and runs before the clock starts. Work you defer to your first `get_move` comes out of your match clock instead.
- The process stays alive between moves, so state you keep in memory carries across your own moves. It does not carry to your next game.
- Your process is suspended while your opponent thinks, so work you leave running between your own moves does not run. Each side has the core to itself while it thinks.
- During your own move one thread is fastest. Threads past the first share the single core and cost you time.
- Games are independent, so two of your games can run at the same time in separate containers. Your agent is never asked for two moves at once.
- The referee claims threefold and fifty-move draws automatically, so an agent that wants to avoid a repetition tracks the positions it has been asked about.

## Wire protocol

The runner handles the wire. It writes one JSON line to your process per move request

```json
{"fen": "...", "time_left_ms": 87500}
```

and reads one JSON line back.

```json
{"move": "e2e4"}
```

- `time_left_ms` is your remaining clock before this move. The increment lands after you move.
- Your colour is the side to move in the fen.
- The first fen you receive is the starting position of the game. Repetition and fifty-move counts begin there.
- Every game starts from a curated opening position that is close to level. The set is not published. Finished games reveal the positions they were played from.

## Your output and your log

- Your own output cannot corrupt the protocol. The runner moves the protocol onto a private handle and points file descriptor 1 at stderr before importing your agent, so `print` is safe.
- Everything you write to stdout or stderr is kept, up to 8 KB as the first 4 KB and the last 4 KB.
- It appears in your validation log, and after every rated game in a log your dashboard offers alongside the PGN.
- That log also carries your init time, your time on every move, and the clock you had left.
- Only your own team can read it.

## Match conditions

| Condition | Rule |
|---|---|
| Time control | 120 s plus 0.5 s per move, per side, on wall time |
| Init budget | 90 s to import your agent, before the clock starts |
| Hardware | one core of an AMD EPYC 9V74, measured at 2.60 GHz. 2 GB RAM. No network. No GPU. Identical for every game |
| Machine | both agents of a game run on the same machine and take the core in turns |
| Filesystem | read-only apart from 256 MB at `/tmp`. `HOME`, `TORCH_HOME`, `HF_HOME` and the other cache paths already point there |
| Scratch | `/tmp` starts empty for every game and is deleted with the game, so use it as scratch space, not as a cache between games |
| Processes | at most 128 processes and threads alive at once. On one core, threads past the first cost you time |

## How a game ends

- An illegal move, malformed output, a crash, running out of memory or missing the init budget loses the game. A move payload over 4 KB counts as an illegal move.
- Losing on time loses unless the other side has no way to mate, and then the game is a draw.
- If both sides fail the game is void. There are no retries within a game.
- Draws follow FIDE rules. A game still running at 600 plies is a draw, and the opening position counts toward the 600.
- The FIDE rules run through python-chess, which covers stalemate, threefold repetition, the fifty move rule and insufficient material.

## Validation

Validation plays two smoke games against a house agent, one as each colour, and publishes the verbatim log. Both games start from curated opening positions.

## Which build plays

- The latest upload that passed validation is the one that plays, in every round.
- A team may upload 10 times a day.
- Uploads close 11 September 11:00. Your last valid build then freezes.
- For eligible teams the frozen build alone plays the final qualification Swiss, which does not open until every submission has finished validating.
