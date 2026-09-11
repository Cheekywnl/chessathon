# Final refinement: frozen candidate and validation

The candidate ZIP is frozen at SHA-256 `908ca85c2fd041d4269bf063f73ff08f20aac11256716ae617b7fc309ef4cb68`. It contains
48,710,010 uncompressed bytes in 114 files, below the
50,000,000-byte limit. Runtime fingerprint:
`78489e1c8647c1311390d837976e24cf53dc4c52821f5d9144052ddddba51d36`.

**Full-clock testing is in progress. An additional 300-500 Elo or a 90% actual
win rate over the penultimate engine has not been established.**

## Exact comparison

The opponent is the penultimate `hour-release` engine, runtime
`029bec79c7f2c75fdd0a1ca5badbee1c9aaa734c5a1675b0e6231871c46c776c`, based on source commit
ce3bdbb260fa714fecf4279f5befe6bce3ba1c4e with documentation commit
5cb6574ca92b900d165242bd97583a788c8881c2. Its already completed 42-win,
15-draw, one-loss result was against the older neural 7eda0fa engine. That older
result is not counted as a new gain here.

The independent comparison uses the actual extracted final ZIP, 120 seconds
plus 0.5 seconds per move, fresh processes for every game, both colours, and
unused opening indices 100-129. Six workers use separate physical CPU cores;
opponent process trees are suspended between moves. The CPU interpreter uses
the competition-pinned dependencies. The final agent selects Numba OPT=2;
the opponent retains its original OPT=3 environment. No harness file changed.

The predeclared sequential test compares 0 versus 50 local Elo with alpha/beta
0.05, at least 20 complete pairs and a cap of 60 games. Already running pairs
are retained after a decision. A capped test can be inconclusive. This is a local
comparison on Windows, not a measurement of the competition's platform rating.

## Selected changes

- Exact bit scans, reusable dense-layer buffers and division kernels, cached
  king-bucket accumulators and static evaluations reduce repeated CPU work.
- Exact pin/check-evasion masks avoid unnecessary attack recomputation. King
  moves and en passant retain full resulting-position checks.
- Root PVS avoids invalid retry windows. Endgame quiescence keeps captures that
  can save a draw, including a reproduced pawn-capture failure.
- The moderately trained broad-teacher width-128 model was selected by games.
  [The model card](HALFKP_MODEL_CARD.md) records the complete lineage and hashes.
- A compact exact KRP-versus-KR policy fixes the reproduced Lucena repetition.
  Its 3,139 entries cover the complete chosen-move/any-legal-defence graph from
  six seed positions, plus colour and file symmetries. This is partial coverage
  of five-piece rook endings. Missing positions return to search. Every lookup
  checks exact material, move legality, the fifty-move window and real-game
  repetition claims. KQR-versus-KR WDL/DTZ files continue after queen promotion.

The partial policy was derived from hash-verified Syzygy files, not learned from
final-test openings. Its generator, seeds, source hashes and full-graph checks
are archived. [The event rules](https://aichessathon.com/docs/rules.md), retrieved
11 September, permit endgame tables at seven or fewer pieces regardless of their
source; these lookups require exactly five pieces. The complete KRP-versus-KR
source table remains offline because it does not fit the submission budget.

## Short selection matches

Each row uses 40 games at 15s+0.2s and 20 colour-swapped openings. These are
selection screens, separate from final testing. Compilation is reused but game
state is reinitialized and checked. Early interrupted batches are archived and
excluded; only completed fixed-worker reruns enter this table.

| Candidate | Opponent | Wins | Draws | Losses | Score | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Optimized original width 128 | Penultimate | 18 | 12 | 10 | 60.00% | Keep foundation |
| Width 64 | Optimized width 128 | 10 | 18 | 12 | 47.50% | Reject |
| Moderate broad teacher 128 | Optimized width 128 | 19 | 13 | 8 | 63.75% | Select model |
| Persistent quiet history | Moderate teacher | 12 | 14 | 14 | 47.50% | Reject |
| Stronger teacher 128 | Moderate teacher | 5 | 16 | 19 | 32.50% | Reject |
| Broad teacher 192 | Moderate teacher | 10 | 14 | 16 | 42.50% | Reject |

All 240 selected-screen PGNs replay legally with complete colour pairs and zero
candidate failure/fallback markers. The effects of separate rows cannot be added
as proven Elo gains, and short-clock results may differ from the full clock.

## Verification

Ruff passes and strict mypy passes all 59 checked source files. The actual ZIP
cold-imported in 37.654s on one physical core, below 90s;
peak process memory was 483,921,920 bytes, below 2 GB. Opening,
middlegame and tablebase smoke moves were legal and within their clocks.

Search parity covers 56 cases, 32 abort cases and 1,944 TT probes; deadline and
cancellation checks pass. The final search/model solves 266/300 WAC positions.
Two random-baseline gate games finish in checkmate. Draw/referee regression,
fifty-move checks, forced-reply checks and draw-saving quiescence regression pass.

The initial final-model Lucena test drew by repetition and was retained as a
failed diagnostic. After the policy fix it checkmated the four-second search
opponent. Four colour/file variants also checkmated an exact-tablebase defender.
All 3,139 policy entries and 5,856 retained opponent-reply edges were checked
against WDL/DTZ; 12,556 symmetry cases and 25,112 fifty-move boundary checks pass.
A separate real-agent check rejects the policy move when history makes it repeat.

Exact fixed-depth benchmarks establish individual CPU savings, not Elo:
bit scans 1.141x; dense scratch/division 1.079x over its parent; static evaluation
cache 1.190x over scratch; check-evasion masks 1.024x; compiler OPT=2 about1.053x
search and 1.119x cold initialization in both timing orders. Each comparison
retained identical tested search states. Noisy early-cache and persistent-static
changes were rejected. Detailed raw measurements and discarded patches are in
[the evidence archive](validation/refinement-final-20260911/SHA256.json).

## Publication

The earlier `submission-refined.zip` is preserved byte-for-byte. The final ZIP
will be delivered as `submission-final.zip`, with a final result summary and
private Git branch `final-release`. The frozen candidate and provenance are
saved there while tests run; completed results follow in a separate documentation
commit after the full-clock audit.

An exploratory 16-game full-clock model attribution match uses unused indices
140-147 on CPU cores 0 and 10. Its opponent has the exact final runtime and assets
except for the previous model weights. It is reported separately and cannot be
pooled with the penultimate-engine comparison. The predeclared primary release
test remains the 60-game comparison above.
