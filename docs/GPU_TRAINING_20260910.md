# GPU HalfKP execution record — 10 September 2026

## Isolation and rules

- Rollback: `05046b28ad9131de8d52b3b991b16601b362167c` in the clean
  `work/baseline-05046b2` worktree.
- Candidate: `halfkp-trial`, private origin
  `https://github.com/Cheekywnl/chess-codex-isolated-20260910.git`.
- The original `C:\Users\cheek\chessathon` checkout and its uncommitted trainer edit
  are preserved. Its shared Git configuration was not changed. This task uses a
  separate bare repository; its shared `upstream` has a disabled push URL.
- Fetched the live agent contract and rules on 10 September. Training on engine
  labels is allowed; published networks and shipped third-party engines are not.
  Each shipped network must originate from this team's random initialization.
  The process is suspended between turns; sandbox resources are one CPU core,
  2 GB RAM, no GPU/network, and a 50 MB unzipped submission limit.
- The baseline passed `uv sync --frozen` and `make gate` before runtime code edits:
  ruff, mypy (28 source files), and two clean checkmates against the random agent.
  GNU Make on Windows uses `SHELL=C:/Program Files/Git/bin/bash.exe`.

## Existing data and environment

- Main CSV: `C:\Users\cheek\chessathon\data\lichess\lichess_eval_large.csv`,
  9,348,699,018 bytes. The original preparation log reports 150,000,000 rows.
  Full independent row accounting and SHA-256 are recorded when preparation ends.
- Source: https://database.lichess.org/#evals, CC0. CSV header `fen,depth,cp`.
  The existing preparation code chooses the deepest evaluation, the first PV,
  filters shallow evaluations, and excludes mate-only records. The historical
  minimum-depth command has not yet been independently recovered; do not invent it.
- Labels are White-relative centipawns. Target is `sigmoid(stm_cp / 400)`.
  Self-play CSVs at `data/selfplay_shards/*.csv` have `fen,mobility,result`;
  result is White's WDL outcome, complemented for Black to move.
- No dataset was downloaded. Existing eight 18.75M-row chunks remain untouched.
- CUDA environment: sibling `work/cuda-env`; Python 3.12.14, torch 2.5.1+cu121,
  NumPy 2.5.2, Numba 0.67.0, chess 1.11.2. RTX 3070 CUDA tensor multiplication passed.
  Neither the original CPU `.venv` nor `uv.lock` was modified.
- Initial free RAM was approximately 7.38 GB, with about 56.8 GB free on C: and
  1.09 TB on D:. GPU use was 598/8192 MiB. Resource samples are included in run logs.

## Deterministic bounded preparation

`tools.halfkp_data` reads one CSV row into a 250,000-row, 128-byte/row buffer
(32 MB). It writes separate train/validation arrays for that shard. Every source
row is visited; invalid records are counted and examples retained. No full list
of FENs or dataset-sized feature matrix is held in RAM.

The split uses BLAKE2b of the first four FEN fields (counters excluded), an eight-byte
digest with personalization `halfkp-split-v1`, interpreted little-endian modulo
1000. Buckets 0–9 are validation. Repeated canonical positions cannot cross the
split, including duplicates in different source files. Shards and rows are shuffled
deterministically each epoch, seed `20260910 + epoch`. An epoch visits every train
shard. All validation shards are held out and evaluated without shuffling.

The first 100k source rows exposed eight boards with more than 30 non-king pieces;
one contains 49. The old fast encoder did not bound its 30-element output writes.
The new encoder checks the count before writing; the legacy helper now has that
guard too. This is a demonstrated out-of-bounds hazard, not proof of the cause of
any particular historical crash. Other malformed boards/labels are rejected and
accounted for in the final manifest rather than silently retained or truncated.

Reference verification command:

```
..\cuda-env\Scripts\python.exe -m tools.halfkp_stream_check --data C:\Users\cheek\chessathon\data\lichess\lichess_eval_large.csv
```

It passed 2,004 feature/target comparisons, 2,004 independent board-mirror checks,
counter-independent splitting, malformed-input rejection and a 137-row roundtrip
across multiple small shards. Training-only factor coalescence differed from the
independent NumPy forward reference by at most 0.00000763 centipawns.

## 100k pilot

```
..\cuda-env\Scripts\python.exe -m tools.halfkp_data --data C:\Users\cheek\chessathon\data\lichess\lichess_eval_large.csv --out data/pilot-cache-v2 --shard-rows 100000 --limit 100000
..\cuda-env\Scripts\python.exe -m tools.train_halfkp_stream --prepared data/pilot-cache-v2 --out data/runs/pilot-128 --epochs 3 --width 128 --batch-size 8192
```

- 100,000 accepted rows, 98,939 train / 1,061 validation.
- Preparation including CSV parsing, split hashing and encoding: 1.384 seconds.
- Training: 296,817 row presentations in three passes; 1.426 seconds total,
  with 214,918 rows/sec at the final epoch report. This tiny cached pilot is an
  optimistic throughput screen; full-shard and validation overhead must be measured.
- GPU utilization sampled at 58%; total GPU memory 1,025 MiB; process RSS 1.039 GB;
  free system RAM approximately 6.99 GB. No OOM or crash.
- Validation probability MSE: 0.020094 → 0.018645 → 0.018180.
- Checkpoint SHA-256:
  `9d1f47c3873cf30c33e7f66dd1002e7c558d2cae937583301008c0ebc6107a8c`.
- Float export SHA-256:
  `1ae9132d1c05acc78e7f1810a1d09d418e22f8bca1811a1d23bd5c5d07817939`.
- At the pilot rate, 150M training rows would take about 12 minutes per pass;
  sustained full-data preparation was subsequently measured at about 100k rows/sec
  (roughly 25 minutes). These are estimates, not completed full-run durations.

## Full run and promotion status

Full preparation includes the existing 150M CSV plus all ten existing self-play
CSVs. `data/full-cache/manifest.json` records accepted/rejected/source counts,
source hashes, shard sizes, split semantics and preparation duration. The full run
uses `--prepared data/full-cache --out data/runs/full-128 --epochs 3 --width 128
--batch-size 8192 --validate-every 10000000`.

Architecture: shared 40,960 × 128 HalfKP transformer, vertically mirrored Black
perspective, side-to-move accumulator first, clipped activations, 32 → 32 → 1 head.
A randomly initialized 640 × 128 piece-square training factor is added to each
king bucket before export; no factor lookup remains in inference. Training starts
from random initialization with seed 20260910. Checkpoints are selected by held-out
loss, with early stopping after four consecutive non-improving validation checks.

All data, float exports, optimizer checkpoints and CUDA files remain ignored local
artifacts. Pilot loss is not strength evidence. No network has been promoted.
Classical `05046b2` remains the rollback opponent until integer inference, package,
correctness and real subprocess match checks succeed. Fresh rule inspection also
found that its book lookup is not restricted to move 20: the submission candidate
now explicitly declines book lookups at move 21 and later.

## Completed full preparation and first run

Every one of the 150,000,000 labelled CSV rows and 56,927 self-play rows was read.
11,629 labelled rows were rejected for more than 30 non-king pieces. Accepted total:
150,045,298, split into 148,547,121 train and 1,498,177 validation positions.
Preparation took 1,699.771 seconds. All source/shard accounting checks passed.
Main CSV SHA-256: `6ce1108f70e3d14213fc1d0b41b024b5c3768a64e98247ab406562ba9d9650ec`.
Prepared manifest SHA-256: `6d63c217336798ec31b4b881f938d4d274eecc587678fad202b3b48e58df22b3`.

The first run presented 180,081,830 training rows (one complete epoch and a partial
second), stopping after held-out loss stopped improving. Duration: 375.845 seconds;
22,556 optimizer steps; best probability MSE 0.01148086245. Checkpoint SHA-256:
`9b6f9f3c23e4a9b37b85595c8320422a8aa998031be5a87b6906edf97da96c57`.
Best float export SHA-256: `2772c8656405de51e9ca89f2fcc3a874b46d3405769e1bf66e4b938a40e03d84`.

The first integer asset is 6,949,254 bytes, with int16 transformer/accumulators and
int8 downstream weights. Its conservative accumulator bound is 15,140, below 32,767.
SHA-256: `cd2c6c5ea9c14f6fb85512e373a2f2b1f2348e49118f96f710d911ec45db0768`.
On 10,000 held-out positions: zero integer-reference and bitboard-inference
mismatches; mean absolute float/quantized error 4.794 cp, p99 14.898 cp, maximum
27.214 cp. Probability MSE on that subset: float 0.01046756, integer 0.01051172.

## First packaged candidate — still unpromoted

The 75% neural blend preserves the classical evaluator for seven or fewer pieces
and bare-king conversions, and adds `mop_up_bonus` to the neural route. Extracting
that helper gave 12,000 exact evaluation matches and ten exact depth-5 move/score/node
matches against `05046b2`. Skipping unused classical work at a 100% neural blend
was separately checked on 30 exact move/score/node comparisons at blends 0/75/100.

The book was reduced from 16,484,048 to 12,500,000 bytes: first keep the deterministic
best entry per key, then drop the lowest-weight keys. There are 781,250 retained
keys and 139,850 dropped keys. Every retained key's maximum was checked. Among
100,010 position probes, all retained responses matched; coverage decreased by 692
of 13,699 original hits. This coverage trade remains subject to real match results.
All original Syzygy files are retained. Capped book SHA-256:
`16a2ab58fb6793e6749a23e1a59ee2e4fc2af5800849c3ba9dafaa94b190f88b`.

The resulting zip was 49,468,870 bytes unzipped; SHA-256:
`02f397a1ff2c6dfc7eacd9c0c4cb05619460df86d95ffdb291711dfe3d2289ce`.
Source/asset audit passed. The extracted zip imported in 10.297 seconds on one
core, reached a measured 207,736,832-byte peak working set in the import/move smoke,
and returned legal opening, middlegame and endgame moves. This is not a full-game
peak-memory claim. Full gate passed; Lucena passed; WAC scored 232/300 at 1s per
position. These are screening results, not strength evidence.

The first real 20s+0.3s, twenty-game screen uses `tools.version_arena`, split into
two independent ten-game batches on cores 2 and 4. Each pair alternates colours;
together they cover all ten built-in openings. Both agent interpreters use the
unchanged pinned CPU environment. The controller suspends the actual Python child
as well as its Windows venv launcher between turns. A busy-counter test proved
that suspending only the launcher is insufficient; the corrected process-tree
controller stopped both counter and CPU-time progress outside turns.

`data/runs/first75-a.jsonl` and `first75-b.jsonl` contain complete game/PGN/log records.
Failed or void games now abort version/SPRT scoring instead of being counted as
draws. The completed screen scored +8 =10 -2 (65%), with ten checkmates and ten
threefold repetitions. Full PGNs replay legally. There were zero candidate runtime
error/fallback markers and no crash, illegal, flag or void outcome. One baseline
log contained a partial ponder-thread exception after its final move before mate;
this diagnostic is retained explicitly in `first75-summary.json`. No network has
been promoted or committed as an asset.

## Continued training and rejected bounded experiment

`--resume` accepts a team checkpoint only after its SHA-256, source-manifest hash,
seed and random-initialization lineage match the adjacent run record. A resumed
pilot reproduced the parent's held-out MSE exactly before taking another step.
This continues our own training; no published network is involved. The lower-rate
continuation uses `--out data/runs/full-128-anneal --epochs 2 --lr 0.0002 --patience 8
--resume data/runs/full-128/best_checkpoint.pt` with the same prepared data and batch
size. It writes separate checkpoints and does not change the candidate under test.

The isolated `no-ponder-trial` passed its gate but failed the Lucena regression by
threefold repetition. It was rejected at that screen. Its patch remains isolated;
the original baseline is untouched. No claim is made that disabling pondering is
safe simply because the live platform suspends opponent-time work.

## Terminal ponder diagnosis and wider testing

The diagnostic above was reproduced from the exact recorded position: after
`d1c2 d2d1q#`, the old worker called `search_root` with no legal moves and raised
`IndexError: list index out of range`. The candidate now checks whether the
predicted reply ends the game before starting a worker. The reproduction test
proves that terminal predictions do not start a thread while ordinary predictions
still do. This is a bounded guard, not the rejected broad removal of pondering.

Match evidence now records exact root-source/book/Syzygy/network hashes, refuses
existing log paths, checks that builds remain unchanged, measures subprocess peak
working sets, replays PGNs, rejects failed outcomes and runtime fallback markers,
and requires unique completed opening/colour pairs. A baseline ponder exception
after its final returned legal move, immediately before the opponent's actual
checkmate, is retained with an explicit qualification. Such post-move diagnostics
cannot change the completed game's result. Tracebacks may be truncated by the
referee stopping the finished process, so not every truncated cause is asserted
to be proven. Candidate errors, earlier errors, subsequent moves and failed
outcomes are explicitly rejected by the tested classification rule.

An additional 100-opening pool was selected before seeing its match outcomes,
using seed 20260910, square-root-weight sampling from the original existing book,
and the original classical evaluator at depth 4 to reject |score| >125 cp. All
positions are distinct and retain at least 26 pieces, drawn evenly from ten opening
families. This is our local pool, not the platform's undisclosed opening set.
Command: `python -m tools.make_opening_suite --baseline ../baseline-05046b2
--out data/runs/openings-20260910.json --count 100`. Pool SHA-256:
`9a99073e107677f0fbef76759c4dcc449b60c29672c5b028d1ffa64d1167fe47`.

The 128-wide anneal completed two full additional epochs (297,094,242 presentations,
37,212 steps) in 600.240 seconds, with best validation MSE 0.01065746171.
Checkpoint SHA-256: `85fe3648e14711f989b85bc590499cbe6b12e5a8aa39262ac0c81cf877234e0a`.
Float SHA-256: `045f8db4f39e92dc1f146d3caf5f42c96d7845a2fdcafb9320e5d420ecd1565e`.
Integer asset: 6,942,764 bytes, SHA-256
`3ab6d135ecdd3b9b24b6c37aace3e9a658c3cc8c6c2093dee1a02edb1a0401a2`.
Its 10,000-position check had zero integer/reference or board-path mismatches;
mean absolute error 5.250 cp, p99 19.344 cp, maximum 41.234 cp. Subset probability
MSE was 0.00941733 float versus 0.00941929 quantized. It awaits real matches.

A separate random-initialization width-64 run (seed 20260912, batch 8192, initial
LR 0.0005, four epochs maximum, patience eight, validation every 20M rows) explores
the measured CPU and book-space tradeoff. Its outputs stay separate in
`data/runs/full-64`; it is not substituted into any live tested build.

## Wider models and promotion test design

The 128-wide annealed asset separately passed the gate, Lucena, 10k quantized
reference, one-core integration and actual extracted-package checks. WAC: 228/300
at 1 second per position. Its isolated worktree is `halfkp-anneal-128`; it is now
playing a fresh 20-game 20s+0.3s screen against the original baseline. The first
asset continues its independent forty-game 120s+0.5s test in `halfkp-trial`, on
opening-pool entries 0 through 19. Both use source commit `2596a86` and distinct,
uncommitted weight assets. Their live runtime files remain frozen.

The width-64 model completed four full passes (594,188,484 presentations, 74,424
steps) in 1,084.811 seconds, with best held-out probability MSE 0.01113301740.
Checkpoint SHA-256: `ef8f4b729acb661a64ad65636a300b77480cd0457744c81573b2006304ed41b2`.
Float SHA-256: `a170e6cb5bffe4dc2ae3f2d6625a8be5c7cdc6cd9b1ed7bbb20a6e2d19e2f1ee`.
The first power-of-two quantization failed the existing error limits: MAE 15.255 cp,
p99 42.361 cp. It was not shipped or tested as a strength candidate. Using the
available int8 range instead (integer hidden scales 122 and 82, output scale
0.22669325843) requires no inference code change and gave MAE 3.665 cp, p99 15.727 cp,
maximum 27.418 cp, with zero integer/board-reference mismatches on 10k held-out
positions. Probability MSE: float 0.01000552, integer 0.01002572.
Accepted integer artifact: 3,334,402 bytes, SHA-256
`ef794958bb77a239461f135fdc36e919f3b7c625aa50a203dd0ae6f8485be8f6`.

The width-64 candidate retains the highest-weight entry for every original book
key (14,737,600 bytes; no key coverage loss), and all Syzygy assets. Its package
is 48,092,037 bytes unzipped, SHA-256
`15a541c6ac6b6acd31d2dae102f7f17ab39cbc7b9b73ae8388007671e892057b`.
The extracted-package smoke imported in 10.423 seconds, peaked at 202,391,552 bytes,
and returned legal moves. At depth 5 on ten fixed positions, it measured about
25.1k NPS at blend 75 and 28.0k at blend 100 during other isolated-core jobs; timing
and tactical screens remain distinct from playing strength.

`tools/sprt_arena.py` now uses a pentanomial constrained maximum-likelihood ratio
with each reversed-colour opening pair as one trial. It evaluates in preselected
opening order even with multiple CPU workers, never repeats an opening as a new
trial, stops only at a completed pair, and retains in-flight pairs. The statistical
definition is [Van den Bergh's GSPRT note](https://cantate.be/Fishtest/GSPRT_approximation.pdf),
equation 1.1; the code derives its one-dimensional Lagrange solve independently.
No chess-engine code or weights were imported from statistical references.
Explicit regularization is 0.001 count per score cell, with at least twenty pairs
before a decision by default. Mathematical constraint/binary/symmetry checks and
a four-game, two-worker real-protocol plumbing test passed; void rejection and
actual process suspension were retested. This statistic does not establish a
platform rating. For the final selected build, the planned fresh test bounds are
logistic local Elo 0 versus 20, alpha=beta=0.05, up to 200 games. The selection
screens will not be pooled into that test.

The tools now subtract the starting FEN's existing plies from the live total cap
without changing `harness/`. Earlier runs used the harness's cap of 600 additional
plies; no completed screen game reached that cap. This distinction has no effect
on their recorded checkmates and repetitions and remains documented explicitly.

## Completed selection screens and final training continuation

The 128-wide annealed model finished its 20-game screen at **+14 =5 -1 (82.5%)**,
with 15 checkmates and five repetitions. The 64-wide model finished **+13 =3 -4
(72.5%)**, with 17 checkmates and three repetitions. Both used 20s+0.3s, all ten
built-in openings, paired colours, one CPU and actual process suspension. Both
passed full PGN/result replay, fixed-build fingerprint, init/memory and runtime-log
audits, with zero candidate error/fallback markers and no baseline diagnostics.
These small selection screens are not pooled into the final sequential test.

The original 128-wide candidate's 120s+0.5s batch A paused after six completed games
because its baseline traceback contained more text than the first observed example.
The final legal move and immediately following mate were verified; the diagnostic
was preserved. Resume begins at pool entry 3, with no replayed or dropped result,
in `first75-ltc-a-resume.jsonl`. The first six results stay in `first75-ltc-a.jsonl`.
Batch B continues independently. Source and asset fingerprints remain unchanged.

A second 128-wide continuation used the same own-training lineage, LR 0.00008,
two full epochs, batch 8192 and seed 20260910. It completed 297,094,242 additional
presentations in 627.767 seconds; best full held-out MSE 0.01058514020.
Checkpoint SHA-256: `3b8fabc0acfe7aa978f8011f66575da8b32d3bdd23c44bf4eaf1b855cdeb61c1`.
Float SHA-256: `70da243032175cb7abbb6c0b703d45176b26df9d07217bd760ef4934c9c6888a`.
Full-range integer asset: 6,936,983 bytes, SHA-256
`dbdf991eff2d615712698e8631bbe40e4801069f7a5723976928ea1721603190`.
On 20,000 held-out positions it had zero reference/board mismatches, MAE 7.991 cp,
p99 20.206 cp and maximum 40.041 cp. Its isolated `halfkp-refine-128` candidate
uses blend 100 in normal middlegames, retaining the classical low-material and
bare-king routes. It passed gate, Lucena, integration, package smoke and WAC
(232/300), and is now in its own 20-game quick screen. It is not promoted.
The package is 49,457,018 bytes unzipped, SHA-256
`a8ae0bf68b04ca7b4270e8ed85bc0f1716bfe9315b586a88db887dc0436b7f25`.

Across the full-data runs, 128-wide training presented 774,270,314 rows and
64-wide training presented 594,188,484 rows: **1,368,458,798 presentations** in total,
including repeat passes over the same training split, not that many unique data
positions. Full-data GPU training consumed about 44.8 minutes; preparation took
about 28.3 minutes. All training is complete pending any evidence that a further
revision is needed; remaining selection is based on actual games.

The final pool uses seed 20260914, 200 distinct FENs and 200 distinct deterministic
book exits, excluding all earlier pool positions/book exits and the ten built-in
screening lines' book exits. This prevents two different starting positions that
converge through book play from being counted as distinct final trials. Command:
`python -m tools.make_opening_suite --baseline ../baseline-05046b2
--out data/runs/openings-final-20260914.json --count 200 --seed 20260914 --cpu 14
--distinct-book-exits --exclude-file data/runs/openings-20260910.json`.
Pool SHA-256: `a41142b1dc37d3a5cfbbee4ae0b7e6e0a41ffb1e76a819a25db775c324e3925d`.
Initial book coverage on these positions: original/all-key book 53/200, capped
book 51/200. Selection was completed before observing any game from this pool.

## Final candidate selected before fresh sequential games

The refined 128-wide blend-100 screen completed at +8 =10 -2 (65%), with clean
candidate logs and legal games. The four completed twenty-game selection scores
were: first128 blend75 65%, anneal128 blend75 **82.5%**, width64 blend75 72.5%, and
refine128 blend100 65%. Select **anneal128 blend75** for the independent final test;
no further parameter or asset changes are made in that worktree. Its short
tournament-clock check has started with a win and a draw; the remaining pair
continues separately, and those selection/check games are not pooled into SPRT.

Selected runtime fingerprint:
`d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c`.
Baseline runtime fingerprint:
`2e507f40e8da1c8b08a89d6c49367460c0ed6ef47fad34a1c688e4fdacd4c5ff`.
The baseline is still original `05046b2`; the candidate uses engine source
`2596a86`, the documented annealed integer asset and 12.5 MB capped book.

Predeclared final command (controller uses the isolated CUDA environment only for
its psutil dependency; every agent runs the unchanged competition CPU interpreter):
`python -u -m tools.sprt_arena --agent ../halfkp-anneal-128
--opponent ../baseline-05046b2 --base-ms 120000 --increment-ms 500
--workers 8 10 12 14 --engine-python <halfkp-trial>/.venv/Scripts/python.exe
--opening-file data/runs/openings-final-20260914.json --max-games 200
--min-pairs 20 --elo0 0 --elo1 20 --alpha 0.05 --beta 0.05
--jsonl data/runs/final-anneal75-sprt.jsonl`.
The first 100 fresh opening pairs are eligible; boundaries are ±2.944438979.
Completed pairs are evaluated in their preselected order. No promotion occurs
until this evidence and the final artifact audit support it.

## Release staging and repeated endgame audit

The separate `halfkp-release` worktree starts from `88c48cd` and copies only the
selected annealed integer network and capped book. Its runtime fingerprint is
exactly the selected `d488f913...`, independently checked against every archive
member. The staged zip is 49,462,799 bytes unzipped, with 537,201 bytes of headroom;
all 96 original Syzygy files (29,874,560 bytes) are unchanged. New runtime files
are `chess_halfkp_int.py` and `weights/halfkp.npz`; changed runtime files are
`agent.py`, `chess_eval.py`, `chess_search.py` and `book/codekiddy.bin`. No runtime
file is removed. The sole `pyproject.toml` change adds the integer module to mypy's
file list; submission dependencies and `uv.lock` remain unchanged.

The selected model's separate four-game 120s+0.5s clock check finished **+2 =2 -0**,
two mates and two repetitions, with legal PGN replay and no runtime diagnostics.
Maximum observed peak working set was 253,558,784 bytes; maximum init was 11.860s.
Those four games are not included in the independent final SPRT.

Windows processor-topology enumeration confirmed physical sibling pairs
`[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]`. Concurrent game
CPUs 2, 4, 8, 10, 12 and 14 therefore use distinct physical cores. CPU 6 is used
for release checks after the selected model's clock check completed.

Release `make gate` passed ruff, strict mypy (44 source files), and two clean
real-protocol mates. A subsequent one-core Lucena repeat drew by repetition,
despite this exact selected build's earlier pass. This failure is retained in
`data/runs/release-endgame.log`; it is not relabeled as a pass. The unchanged
baseline's corresponding run passed. A diagnostic compared 21 legal low-material
positions at depth 6: both engines matched exactly in move, score, node count and
every ranked root score. Traced reruns of the original endgame procedure then
converted Lucena to checkmate for both candidate and baseline, with full legal
PGNs retained. The candidate trace took 57 plies. No engine policy was changed.
This establishes timing sensitivity of the existing endgame screen, not a
guarantee of conversion under every timing condition. The initial draw remains a
reported limitation, alongside the passing runs and exact fixed-depth evidence.

The standalone model card and compact training provenance now record the exact
selected lineage, source accounting, training commands and quantization checks.
The original data and float/optimizer checkpoints remain local; no weight has
yet been committed or promoted. The final sequential test and release checks
continue with frozen runtime assets.
