# Chessathon: continue on this machine (RTX 3070, i7-10700K)

You're picking up an in-progress chess engine for AI Chessathon (aichessathon.com), a live
competition. Another Claude Code session has been working this same repo tonight on a laptop
and just handed off the NN-training track to you because this machine has a real GPU. The
deadline is real and close. Read this whole document before touching anything.

## Do this first, before any code

1. Fetch and read **https://aichessathon.com/docs/agent-contract.md** and
   **https://aichessathon.com/docs/rules.md** directly — they are canonical and change; this
   document is a snapshot, not the source of truth. In particular, before doing ANY training
   work, be certain you understand the rules around what counts as "shipping an engine" vs
   "training data" vs "a network you trained yourself" — getting this wrong risks
   disqualification, which is worse than not training anything at all.
2. Read this repo's own `AGENTS.md` / `CLAUDE.md` at the root — platform constraints, the
   contract, style rules, and "Do not edit `harness/`."
3. Read `viewer/activity_log.jsonl` (tail it, don't read the whole thing) for a chronological
   narrative of tonight's session: what was tried, what worked, what was discarded and why.
   This matters — don't re-attempt things already tried and killed (counter-move heuristic:
   regressed, real A/B 40%; Texel tuning: tried twice, no validated gain; magic bitboards for
   sliding-piece attacks: correct but zero end-to-end benefit, reverted — don't redo it).
4. Check the actual current time against the deadline yourself (`date`) — as of handoff it was
   **Thu Sep 10 08:22 BST 2026**, deadline is **Sep 11, 11:00** (per AGENTS.md — verify against
   the live rules page). That's roughly 26 hours at handoff time, but confirm freshly.

## The rules boundary for anything you build (read the source docs, this is a summary)

- Third-party engines are banned inside the submission — Stockfish, Lc0, Maia, any wrapper or
  port of one. Your moves come from code your team wrote.
- **Training on positions an existing engine labelled is explicitly allowed.** The line is
  starting from or fine-tuning a *published network* — that counts as shipping it, which is
  not allowed. Any network you ship must be trained by your team from a random initialization.
- Model weights (`.onnx`, `.safetensors`, `.pt`) are fine to ship — they're not "binaries."
- A shipped lookup table that answers positions past the opening counts as an engine, not
  training data. Endgame tablebases are the explicit exception. This doesn't affect a real
  trained network doing real inference, only naive "ship a giant position->move table" schemes.
- You must be able to explain how anything you ship was built, including training, if asked —
  finalist teams walk through this. Keep the process legible: clean commits, honest docstrings,
  no obfuscation.
- The competition sandbox itself has **no network access, 1 CPU core, 2GB RAM, no GPU** — this
  machine's GPU is for *training only*. Whatever you ship has to run fast on that constrained
  sandbox at inference time. Keep that in mind when sizing any network.

## What's already shipped and validated tonight (don't redo)

Pull `main` — it's all there. Highlights, most recent first:
- Several verified search speed optimizations (bit-count, an `insufficient_material` bitboard
  rewrite that was ~23% of search time, redundant Zobrist-hash deduplication). Real measured
  payoff: current build vs the pre-optimization baseline scored **67.5% in a real A/B**
  (`tools/version_arena.py`, 20 games, 10 openings, 20s+0.3s clock — cp-e5 flagged this clock
  is faster than the real 120s+0.5s and a small confirming batch closer to real time control
  would be good insurance before leaning hard on that exact number; consider running one).
- `tools/version_arena.py` — real-protocol A/B arena across diverse openings (two deterministic
  engine builds from the bare start position only produce ~2 distinct games no matter the count;
  this fixes that).
- `tools/sprt_arena.py` — proper Sequential Probability Ratio Test, the actual methodology real
  engine devs use (fishtest/OpenBench/cutechess-cli), not a fixed game count with an eyeballed
  confidence interval. Stops as soon as there's enough evidence either way.
- `tools/generate_training_data.py` — self-play + real-game data generation for training,
  streams/flushes incrementally now (was buffering everything and losing partial progress).
- `tools/prepare_lichess_eval.py` — streams the Lichess open evaluation database
  (database.lichess.org, CC0-licensed, 409M+ Stockfish-evaluated positions) into a sampled CSV
  without downloading the full ~22GB file. Verify the exact download URL yourself against
  database.lichess.org rather than trust one hardcoded anywhere — an earlier automated fetch of
  that page returned two different paths for the same file (a summarization glitch, not a real
  ambiguity), so don't propagate a possibly-wrong URL forward.
- `tools/train_nnue.py` — trains a small value network (currently 768 input features -> 256 ->
  32 -> 1 output, interpreted as centipawns) on either self-play WDL data or Lichess cp data,
  freely mixable, auto-detected by CSV header.

## The one real bug you need to fix before scaling up training — do this first

`tools/train_nnue.py`'s `load_dataset()` currently materializes the **entire** dataset as a
dense float32 feature matrix in memory before training starts. At small scale (1M rows, ~3GB)
this is fine. At 20M rows it needs **61.5GB** — it was launched against 32GB of RAM on the
laptop tonight and was actively swapping before being killed. Each position has only ~32
non-zero features out of 768 (a sparse one-hot board encoding) being stored densely, and worse,
the *entire dataset's* dense form gets built upfront instead of just what one training batch
needs.

**Fix it properly, not with a workaround**: keep FENs and targets as lightweight lists (cheap —
about 1.4GB for 20M rows of text+float), and only call `fen_to_features()` for one batch's
worth of rows at a time, inside the training loop, not upfront for the whole dataset. This is
standard practice (a `Dataset`/generator that yields batches) and scales to any dataset size
with roughly constant peak memory. Verify it against the existing (correct) dense-loading
behavior on a small sample before trusting it at scale — same discipline as everything else in
this repo: prove new code produces identical results to old code before assuming it does.

## The actual mission

The user's bar has been raised: **push hard toward real strength, genuinely competitive play —
they've named 2500 Elo as the target.** Be honest with them about what's achievable in the time
you have, but don't let that honesty become an excuse to coast — burn the ocean. Use whatever
*legal* approach gets there fastest: a trained value network is the most promising untried
lever (the current eval is hand-crafted material+PST+structure, already Texel-tuned twice with
no gain — a trained network is a fundamentally more expressive function class, and it's what
actually took a comparable hobbyist engine, Serendipity, from mid-tier to CCRL 3500+: "several
commits with gains of over 100 Elo" right after adding NNUE, per its own author). But don't
fixate on NN-or-nothing — if you find a better lever (search algorithm improvements, a
genuinely better training signal, anything), pursue it. The goal is engine strength, not a
specific technique.

Whatever you build, it has to actually run fast enough on the competition sandbox (1 core,
2GB RAM, no GPU) — a network that's too slow to call every search node at 120s+0.5s is worse
than the current eval, however good its raw prediction accuracy is. Budget real time to design
and validate a fast inference path (in this repo's style: numba-jitted, quantized if needed) —
this is not a detail to leave for later, it determines whether any of this is shippable at all.

## Non-negotiable discipline (this repo's own hard-won rules tonight)

- **Never trust "should be better/faster" — verify.** Every real change tonight was validated
  by either bit-for-bit-identical output proof (pure speed/refactor changes) or a real A/B via
  `tools/version_arena.py` / `tools/sprt_arena.py` (behavior changes). One optimization
  (magic bitboards) was *correct* and *measurably faster at the function level* but delivered
  zero end-to-end benefit and got reverted once that was properly measured — a real function-
  level win doesn't automatically mean a real system-level win. Measure the thing you actually
  care about, not a proxy for it.
- `make gate` and `uv run python -m tools.endgame_regression` must both pass clean before any
  commit that touches the engine.
- `submission.zip` has a hard 50MB unzipped cap. Check `make zip`'s size warning after adding
  any data (including model weights) — do not blow the budget.
- Discard failed experiments without hesitation, log *why* in `viewer/activity_log.jsonl`
  (`uv run python viewer/log_activity.py "<message>" <status>`) the same way tonight's session
  did for both wins and failures. This log is genuinely useful institutional memory, not
  ceremony — use it.

## Permissions and workflow

- **Push directly to `main` on GitHub** (`git push origin main`) — you have full permission,
  same as tonight's session. Commit real, validated progress; don't leave uncommitted work
  sitting around.
- **You have full permission to use this computer and its files** for anything this task needs.
- Push meaningfully often (not every tiny experiment, but every validated real change) so
  progress is visible and recoverable — this matters given the deadline; a build that only
  exists locally when something goes wrong is a build that's lost.
- **Neither you nor the laptop session has access to the actual aichessathon.com dashboard** —
  only GitHub. The user uploads `submission.zip` to the platform manually. Keep GitHub current
  and tell the user clearly when something is ready to upload; don't assume the dashboard
  reflects your latest commit until they've done that step.
- **Coordinate with the other active session(s)** — use `ListAgents` to see who's around
  (there was a laptop session and a peer session "cp-e5" both active on this repo tonight) and
  `SendMessage` before touching shared hot files (`chess_search.py`, `chess_eval.py`,
  `agent.py`, `chess_movegen.py`) if there's any chance of overlapping with concurrent work —
  same working directory conventions may not apply across machines, but the courtesy of
  flagging what you're about to touch avoids wasted work and merge pain. Introduce yourself and
  say what you're picking up.

Go. Read the two rules URLs and this repo's own docs first, actually — not skimmed. Then move
fast.
