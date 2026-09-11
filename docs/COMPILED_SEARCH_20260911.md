# Compile the team's own search, 11 September 2026

Candidate only; the released neural engine 7eda0fa remains on main. The user's
target is at least 90% actual wins against that engine. The short comparison completed at 17 wins, 2 draws and 1 loss: 85% actual wins,
90% score. All 20 games at 20s+0.3s, using paired opening indices 172-181, are kept.
This is a promising short result, not yet the user's 90% actual-win objective.

## Change

The team's existing negamax, PVS, quiescence, pruning, move ordering, and hybrid
evaluation are compiled together with Numba. This removes Python calls and boxed
objects from the recursive path. The Python root retains draw scans, aspiration
windows, book/tablebase choices, and the real game history. Shared constants keep
both implementations aligned. The Python recursion remains a classical fallback
if the network is unavailable. No third-party engine code or compiled binary ships.

A fixed array transposition table replaces tuple slots, preserving collision
replacement and mate-distance rules. Its default 2**21 entries use 39,845,888 bytes.
Every recursive caller restores the child repetition count before propagating an
abort flag. Only the root boundary raises TimeUp. A periodic monotonic-clock check
also observes worker cancellation under the GIL. The source and runtime compile
inside initialization; no native cache is packaged.

The released width 128 integer model is byte-identical (SHA256
3ab6d135ecdd3b9b24b6c37aace3e9a658c3cc8c6c2093dee1a02edb1a0401a2).
All 96 Syzygy files remain. The preceding forced-reply/terminal draw corrections
and exact gzip book storage are retained. There is no additional training or
dataset change in this candidate.

## Correctness and local performance

- 56 fixed-depth 6 search comparisons across blends 0/75/100,
  ordinary positions, 19 recorded draw histories, castling, en passant, promotions,
  Lucena, stalemate and fifty-move terminals: identical scores, moves, root ranks,
  nodes, every table slot, killers, history, and repetition counts.
- 84 iterative-deepening/persistent-table comparisons, depths 1-7 across three
  successive positions in four openings: identical complete state and decisions.
- 1,944 table normalization/replacement probes include uint64 boundary keys,
  mate-score thresholds, collisions, flags and different plies.
- 32 exact abort-state comparisons against the Python reference at prescribed
  node limits. No incomplete ancestor stores a bound; repetition state is restored.
- Ten 50ms deadline probes returned in at most63.0ms;
  event cancellation returned in78.0ms.
- 40,000 direct hybrid evaluation comparisons over 10,000 held-out positions and
  four blends: zero mismatches. The inherited integer-kernel reference checks also
  apply because the network and inference kernel are unchanged here.
- make gate passes (ruff, strict mypy, two legal smoke-game mates). The final
  tools-only lint/type pass covers 55 files. Lucena reaches checkmate.
- 3,468 referee comparisons, 26 recorded roots, nine fifty-move roots, four forced
  reply mirrors and four tablebase conversions pass. The separate terminal suite
  passes 992 referee states and 502 terminal leaf comparisons.
- Full one-second WAC: 273/300. This is a tactical screen, not an Elo measurement.
- Alternating-order depth 6 comparisons total44.470s Python vs
  7.045s compiled (6.31x).
  The blend 75 subset measures5.99x. Endgame-heavy samples inflate the overall
  ratio relative to middlegames. No hardware-independent speed claim is made.

The earlier isolated width 256 prototype passed 15 depth 6 comparisons and measured
4.17x after warm-up. Its first run included a signed/unsigned entry-signature
compilation in one timing and was superseded by the warmed alternating-order run.
The prototype's initial table-threshold boundary mismatch was corrected before
integration; the 1,944 boundary probes above cover it.

## Package and next decision

The exact extracted package passes source/permitted-assets and legal smoke checks:
44,993,851 unzipped bytes in 109 files,
42.340s one-core local import, and
449,294,336-byte peak working set. These are local measurements;
the platform's upload validator remains the authority for its CPU and environment.

- ZIP SHA256: d05f01a315b7e8e8a8a9496319febda5f6cb86ec83f71754bef788d402456080
- Runtime SHA256: 7b70d2ba4e896ec9f699e0d098aaeaa7c2204502c847d273d2b5d8c93153cba8
- Opponent runtime SHA256: d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c

The short screen uses five distinct physical CPU cores and suspends the opponent
between turns. Only >=70% score and >=60% actual wins clears its preliminary gate;
that is not the 90% objective or permission to promote a weak result. No long
tournament is justified by correctness or speed evidence alone.


## Completed short comparison

All 20 games and 10 colour pairs finished: **+17 =2 -1, 90% score, 85% actual wins**.
Seven openings were won with both colours, two yielded a win and a draw, and one
was split. Eighteen games ended by checkmate; two by repetition. Every PGN replays
legally, the candidate/opponent fingerprints remained fixed, and neither engine
logged any failure/fallback diagnostics. Maximum recorded import was 43.578s and
peak working set was 595,644,416 bytes.

Both drawn games were recognized as draws in the final search; neither retained
an evaluation of at least +150cp. Seven logged root scans completed. This is a
small opening sample and does not establish a platform rating or a guaranteed
85%/90% future result. It clears the preliminary promising gate, while falling
one observed win short of the user's 90% actual-win target in this sample.

No long tournament is started yet: a separate accumulator-cache experiment is
being checked for further speed while preserving this frozen candidate. Main and
the recommended upload remain the previously validated 7eda0fa release.
