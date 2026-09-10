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
Classical `05046b2` remains the safe submission until integer inference, package,
correctness and real subprocess match checks succeed.
