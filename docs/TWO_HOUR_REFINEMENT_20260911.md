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
Its paired comparison against the exact corrected starting engine is in progress.
Search and new-training experiments remain isolated until their tests finish.

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
