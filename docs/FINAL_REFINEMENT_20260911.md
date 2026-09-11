# Final refinement: frozen candidate and validation

The candidate ZIP is frozen at SHA-256 `4d3086d962a7a1760a0e7388fe31f267937c893542693d6a26eb4d67c4712110`. It contains
45,431,701 outer-uncompressed bytes in 114 files. Counting every nested
ZIP/NPZ and gzip archive gives 48,999,996 bytes,
leaving 1,000,004 bytes under the 50 MB limit. Runtime fingerprint:
`99807db7aef6e128c8b1480b5baa5e46a7df821951ea2c87f47785aaec9a8900`.

**The larger-book full-clock test finished at 29 wins, 20 draws and 7 losses
(69.64% score). Its predeclared SPRT accepted H1 after 23 pairs; all 28 started
pairs were retained. All 56 PGNs replay legally with zero runtime failures.
The exact corrected smaller-book ZIP finished 20 games at 15s+0.2s: 10 wins, 6 draws and 4 losses (65% score). This small test was statistically inconclusive; all games replay legally and neither engine logged a runtime failure.
An additional 300-500 Elo or a 90% actual
win rate over the penultimate engine has not been established.**


## Upload-size correction and platform status

The first archive failed platform validation: its gzip opening book expanded
from 7,994,197 to 12,500,000 bytes, bringing the platform's count to 53,215,813.
Our original audit counted only the outer ZIP. That audit was insufficient.
It now counts nested ZIP/NPZ and gzip archives; synthetic nested-format
regressions and the actual corrected artifact pass.

The replacement is **submission-final-fixed.zip**, also copied to
submission-final.zip. It retains the 294,743 highest-frequency opening keys in
a 4,715,888-byte uncompressed book. Every retained entry is byte-identical;
missing keys use normal search. All non-book runtime files, the trained model
and endgame tables are unchanged. On the 300-opening pool, book hits change
from 84 to 67. This changes some opening decisions, so the prior larger-book
full-clock result is supporting evidence, not an exact corrected-ZIP result.

The actual corrected ZIP passed a fresh one-core import and legal-move check.
A separate completed 20-game fresh-process test at 15s+0.2s on unused opening indices
160-169 compared this exact corrected ZIP against the penultimate engine. Its result
was 10 wins, 6 draws and 4 losses; all 10 colour pairs were retained. Maximum recorded
initialization was 62.75s and peak memory was 484,732,928 bytes. All six draws were
repetitions; every logged draw scan completed.

The user supplied a subsequent platform log confirming the corrected archive's
size check passed at 45,431,701 bytes. Docker then failed at
`FROM aichessathon/agent-base:latest` with pull-access denied, before importing
our agent. Platform execution has therefore not yet been validated; the base
image was unavailable to that builder. The user subsequently reported that the
organisers confirmed their error. A successful platform validation log has not
yet been supplied here.

An additional archive audit compared this ZIP with both previous releases:
all CRCs passed, paths were portable, no duplicate or symbolic-link entries
were present, and all Python imports were local, standard-library or supplied
by the platform. `tools.portable_submission` can rebuild the same 114 payloads
with normalized ZIP metadata, and verifies their byte identity. The resulting
diagnostic `submission-final-rebuilt.zip` has SHA-256
`c376a7005be604836c282b93d35a5186e33f0d552fadcf8db676d75c37bfa0da`;
it changes no engine or asset bytes and is not a claimed Docker-image fix.

## Exact comparison

The opponent is the penultimate `hour-release` engine, runtime
`029bec79c7f2c75fdd0a1ca5badbee1c9aaa734c5a1675b0e6231871c46c776c`, based on source commit
ce3bdbb260fa714fecf4279f5befe6bce3ba1c4e with documentation commit
5cb6574ca92b900d165242bd97583a788c8881c2. Its already completed 42-win,
15-draw, one-loss result was against the older neural 7eda0fa engine. That older
result is not counted as a new gain here.

The 120-second comparison uses the pre-correction extracted ZIP with the larger
opening book, 120 seconds
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

Ruff passes and strict mypy passes all 60 checked source files. The corrected ZIP
cold-imported in 47.860s on one physical core, below 90s;
peak process memory was 483,540,992 bytes, below 2 GB. Opening,
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
pooled with the penultimate-engine comparison. That attribution batch was interrupted after eight saved games to free its
workers for corrected-package validation following the size rejection. It is
retained as interrupted research and is not a completed final strength result.
