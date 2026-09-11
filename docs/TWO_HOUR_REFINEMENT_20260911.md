# Additional refinement window, 11 September 2026

The starting engine is the corrected c1b6dc4 submission, runtime fingerprint
`99807db7aef6e128c8b1480b5baa5e46a7df821951ea2c87f47785aaec9a8900`.
Its ZIP SHA-256 is `4d3086d962a7a1760a0e7388fe31f267937c893542693d6a26eb4d67c4712110`.
The budget is 50,000,000 fully expanded bytes; the starting artifact uses 48,999,996.
The user authorized additional work from 07:09 to 09:09 UTC, targeting another
200 Elo or more. Targets are not measured results.

First selection experiments compare network blend 90%, guarded adaptive null
search, and bounded history with promotion ordering against that exact starting
engine. Each uses 40 games at 8s+0.12s, with colour-swapped opening indices
180-199 from the existing reproducible pool. The same positions across experiments
make comparisons easier; results from different experiments must not be pooled.
Private selection workers reuse compilation and reset all game state. Final
confirmation will use fresh processes and separate openings.

Two exact arithmetic-loop alternatives were discarded: their hidden activations
and outputs matched all 4,000 cases, but median dense-loop times were 0.840s and
1.236s versus the current 0.820s. This is a timing result, not an Elo measurement.
Further work includes broader exact KRP-versus-KR policy coverage, with independent
generated seeds, WDL/DTZ checks, preservation of all existing policy entries, and
the same root repetition/clock guards. No third-party engine or network ships.

The earlier release's completed tests are archived separately in
`validation/refinement-final-20260911`: larger-book full-clock 29W/20D/7L,
corrected exact-ZIP quick 10W/6D/4L. Neither is a gain measured during this new window.

## Verified foundation checkpoint

Dense inference now traverses contiguous transposed weights and skips zero clipped
activations. The own-team trained 128-wide network is unchanged. All 4,000 dense
cases (3,000 real positions and 1,000 random activation pairs) have exactly the same
hidden layers and output. An independent full-refresh oracle checks 105,643 positions
and 50,111 legal accumulator transitions with zero mismatches.

Full-search ABBA benchmarking on 20 positions through depth nine produced exactly
the same moves, scores, 4,147,709 nodes, transposition-table contents and histories
in every run. Combined search times improved from 36.119s to 26.930s, a 1.3412x
throughput ratio. This is measured speed, not a measured Elo gain.

The exact KRP-versus-KR policy grows from 3,139 to 9,103 positions while preserving
every original entry. All defensive replies within this material class remain in
the verified policy; transitions use the retained complete smaller/KQR tables.
The independent oracle checked 36,412 symmetry cases, 72,824 fifty-move boundaries
and 6,663 reply edges. Twenty complete agent games against exact DTZ defence all
finished in checkmate, including sixteen newly covered nonpromotion positions.
The history-aware repetition override also passed. The rook asset SHA-256 is
`d7c1fa626fcdaafe9deb7ce2762a636a698c91d3820219e65f7bc632d485fde0`.

The foundation passed 277/300 WAC positions at one second each, the Lucena
regression, and both required five-second random-opponent gate games by checkmate.
Its completed quick paired comparison against the exact corrected starting engine
is recorded below. The search and training variants stayed isolated throughout
selection; their completed results and rejection decisions follow.

Initial 40-game, 8s+0.12s selection results, all against the exact starting engine:

| Experiment | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Neural blend 90% | 7 | 13 | 20 | 33.75% |
| Guarded adaptive null move | 14 | 16 | 10 | 55% |
| Bounded history and promotion ordering | 12 | 18 | 10 | 52.5% |
| Neural blend 60% | 13 | 14 | 13 | 50% |
| History plus guarded null move | 15 | 14 | 11 | 55% |
| Revised pruning margins and PV protection | 15 | 16 | 9 | 57.5% |

All six logs pass legal PGN replay and runtime evidence checks. The first experiment
crossed the rejection boundary; the others were statistically inconclusive.
These selection results do not establish the requested 200 Elo target.

## Final candidate selection

Reusing a 32-element int32 output buffer for intermediate dense sums avoids an
allocation on every evaluation. All 4,000 outputs and hidden layers still match.
A second independent ABBA search comparison against the sparse foundation passed
all move/score/node/table/history checks and measured another 1.0713x throughput.
The final candidate's 105,643-position accumulator oracle check also passed.

The second selection round used 20 new balanced, colour-swapped openings at
10s+0.15s. The generator froze positions before candidate outcomes were seen.
Missing legal replay metadata initially stopped all three runs before any games;
the metadata was repaired and all 80 positions replayed to their unchanged FENs.
Both failed attempts and corrected generator/positions are retained.

| Candidate | Opponent | Wins | Draws | Losses | Score |
|---|---|---:|---:|---:|---:|
| Sparse foundation and expanded rook policy | Exact corrected current bot | 17 | 11 | 12 | 56.25% |
| Deeper teacher network | Exact corrected current bot | 13 | 12 | 15 | 47.5% |
| Revised pruning plus sparse foundation | Sparse foundation | 11 | 17 | 12 | 48.75% |

All three were statistically inconclusive. The new training and pruning variants
are not selected. The final model and opening book remain exactly those of the
corrected current submission; improvements are arithmetic speed and exact rook
coverage. No results from different candidates or opponents are pooled.

The additional network was trained from our existing checkpoint on 6,870 accepted
labels from an independently verified offline Stockfish teacher at 48,000 nodes,
using stable position-hash train/heldout partitions. Teacher validation loss
improved, but the match result did not support deploying it. Training provenance,
metrics and hashes are archived; neither the teacher nor its executable, raw
training data or checkpoint ships in the ZIP.

The frozen candidate ZIP contains 114 permitted files, 41,332,226 compressed bytes,
45,455,743 outer unzipped bytes, and **49,042,751 fully recursively expanded bytes**.
The margin under 50,000,000 is 957,249 bytes. Its SHA-256 is
`2b25f386bcb7c4a795c43449cdc8c937f66f09c97763b5d56c181f4661b65719`;
runtime fingerprint `dc545883006320731603675f07c9f75263a60268fb3cc4be7ed9234498204adf`.
The actual ZIP passed a fresh one-core import in 43.88 seconds, used 491,618,304
bytes peak memory, and returned legal book, search and tablebase smoke moves.
Remote organiser validation remains the authority for upload acceptance.

Final confirmation is predeclared as 48 games (24 colour pairs), 120s+0.5s,
eight independent one-core workers, fresh processes every game, original referee
and runner, 600 total plies, and unseen opening indices 20-43. The hypotheses are
0 versus 50 Elo with 5% error bounds and at least 20 completed pairs; all in-flight
pairs are retained. Opponent: the exact corrected starting ZIP, runtime `99807db7...`.
This confirmation is complete; all 24 predeclared pairs are retained below.

## Frozen candidate verification

A direct final-versus-starting-engine ABBA benchmark confirms the combined speed
gain: 1.4776x throughput (47.8% more nodes per second), with
exactly identical moves, scores, nodes and search-table/history contents across
20 positions through depth nine. Every run visits 4,147,709 nodes. The before
runs take 15.136s and 13.344s; final runs take 9.576s and 9.698s. Absolute timings
depend on machine load; both test orders favour the final implementation.

The final frozen payload passes the full repository Ruff and strict mypy gates
(61 source files), plus two fresh five-second gate games by checkmate. Private
screening helpers have been archived outside executable source, so the final
lint/type checks have no temporary tool exclusions. The original harness is
unchanged. The full-clock result is recorded below.


The final draw-regression run passed 3,468 referee comparisons, detected all 26
recorded draws, and passed nine fifty-move and four forced-reply root checks.
Exact tablebase conversions and forced-move history recording passed; maximum
draw-scan time was 4.34ms. Git object verification confirms all 114 runtime blobs
match the frozen ZIP and every archived evidence file matches its SHA manifest.
Fresh canonical rules and contract copies fetched at 08:12 UTC remain consistent
with the selected package. The local test machine is an i7-10700K; the platform's
published CPU is an AMD EPYC 9V74, so local timings are not remote measurements.


The evidence auditor now also rejects the rook-policy loader's specific
"unavailable" warning, which was not covered by the generic model/tablebase
failure markers. A real missing-asset import reproduced that warning and the
auditor rejected it; clean logs remain accepted. No such warnings occur in the
existing selection or completed final-game records. This changes validation
only; the frozen runtime and game protocol are unchanged. Ruff and mypy passed.

## Completed final confirmation

The exact final ZIP scored **13 wins, 27 draws and
8 losses in 48 games**, **55.21% score**, against the exact
corrected current engine (c1b6dc4), at 120s+0.5s. Every game used a fresh process,
one CPU core and the unchanged referee. All 24 opening pairs and all in-flight
games are retained. Paired GSPRT outcome: **inconclusive: opening/game cap**;
final log-likelihood ratio 0.679013.

All 48 PGNs replay legally. There were no flags, crashes, illegal moves, model
load failures or fallback diagnostics. Maximum recorded initialization was
56.50s; peak memory was
498,843,648 bytes. Both runtime fingerprints
stayed unchanged. The final frozen payload scored 277/300 WAC
and passed the Lucena regression. These final checks ran on CPU0 only after its
last assigned match pair completed, avoiding CPU contention with that worker.

The draw audit finds 27 draws and 0
incomplete logged claim scans. 3 draws finished through
exact tablebase choices; 0 ended with the
candidate's final search still claiming at least +150cp. Raw details are retained.

The descriptive logistic conversion of this score is **+36.3 local comparison
Elo**. This is a small, related-engine match estimate, not a platform rating or a
confidence bound. Neither the 200 Elo target nor the 500 Elo aspiration is
established. The 47.8% search-throughput increase is a separate timing result.
Nine selection experiments completed 360 games before these 48 confirmation
games; their different candidates, clocks and opponents are not pooled.

Upload the separately built submission ZIP, not the GitHub repository archive.
The baseline corrected ZIP remains available as a rollback. The organiser's
remote build/validation log is the authority for platform acceptance; no remote
validation success has been supplied for this new ZIP.

The sole stalemate was independently checked after the match with the public
seven-piece Syzygy service. Black's Rxc7 was the only drawing move; all sixteen
alternatives lose, according to the [oracle query](https://tablebase.lichess.ovh/standard?fen=8%2Fp1R2r2%2FKp6%2F1P6%2F3k4%2F8%2F8%2F8%20b%20-%20-%206%2054).
The runtime itself found the defence by search (depth21,
score-20); it never queried a remote service. Raw oracle response, exact FEN and
the distinction between the root and successor side-to-move categories are
archived in twohour-stalemate-review.json. This is post-match diagnosis and was
not used to train or change the frozen candidate.
