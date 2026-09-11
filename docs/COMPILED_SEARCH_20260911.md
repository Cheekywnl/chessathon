# Compile the team's own search, 11 September 2026

Candidate only; the released neural engine 7eda0fa remains on main. The user's
target is at least90% actual wins against that engine. No playing-strength result
exists for this candidate yet. A predeclared20-game,20s+0.3s paired comparison will
use untouched opening indices172-181 from the recorded pool, with all games kept.

## Change

The team's existing negamax, PVS, quiescence, pruning, move ordering, and hybrid
evaluation are compiled together with Numba. This removes Python calls and boxed
objects from the recursive path. The Python root retains draw scans, aspiration
windows, book/tablebase choices, and the real game history. Shared constants keep
both implementations aligned. The Python recursion remains a classical fallback
if the network is unavailable. No third-party engine code or compiled binary ships.

A fixed array transposition table replaces tuple slots, preserving collision
replacement and mate-distance rules. Its default2**21 entries use39,845,888 bytes.
Every recursive caller restores the child repetition count before propagating an
abort flag. Only the root boundary raises TimeUp. A periodic monotonic-clock check
also observes worker cancellation under the GIL. The source and runtime compile
inside initialization; no native cache is packaged.

The released width128 integer model is byte-identical (SHA256
3ab6d135ecdd3b9b24b6c37aace3e9a658c3cc8c6c2093dee1a02edb1a0401a2).
All96 Syzygy files remain. The preceding forced-reply/terminal draw corrections
and exact gzip book storage are retained. There is no additional training or
dataset change in this candidate.

## Correctness and local performance

- 56 fixed-depth6 search comparisons across blends0/75/100,
  ordinary positions,19 recorded draw histories, castling, en passant, promotions,
  Lucena, stalemate and fifty-move terminals: identical scores, moves, root ranks,
  nodes, every table slot, killers, history, and repetition counts.
- 84 iterative-deepening/persistent-table comparisons, depths1-7 across three
  successive positions in four openings: identical complete state and decisions.
- 1,944 table normalization/replacement probes include uint64 boundary keys,
  mate-score thresholds, collisions, flags and different plies.
- 32 exact abort-state comparisons against the Python reference at prescribed
  node limits. No incomplete ancestor stores a bound; repetition state is restored.
- Ten50ms deadline probes returned in at most63.0ms;
  event cancellation returned in78.0ms.
- 40,000 direct hybrid evaluation comparisons over10,000 held-out positions and
  four blends: zero mismatches. The inherited integer-kernel reference checks also
  apply because the network and inference kernel are unchanged here.
- make gate passes (ruff, strict mypy, two legal smoke-game mates). The final
  tools-only lint/type pass covers55 files. Lucena reaches checkmate.
- 3,468 referee comparisons,26 recorded roots,nine fifty-move roots,four forced
  reply mirrors and four tablebase conversions pass. The separate terminal suite
  passes992 referee states and502 terminal leaf comparisons.
- Full one-second WAC:273/300. This is a tactical screen, not an Elo measurement.
- Alternating-order depth6 comparisons total44.470s Python vs
  7.045s compiled (6.31x).
  The blend75 subset measures5.99x. Endgame-heavy samples inflate the overall
  ratio relative to middlegames. No hardware-independent speed claim is made.

The earlier isolated width256 prototype passed15 depth6 comparisons and measured
4.17x after warm-up. Its first run included a signed/unsigned entry-signature
compilation in one timing and was superseded by the warmed alternating-order run.
The prototype's initial table-threshold boundary mismatch was corrected before
integration; the1,944 boundary probes above cover it.

## Package and next decision

The exact extracted package passes source/permitted-assets and legal smoke checks:
44,993,851 unzipped bytes in109 files,
42.340s one-core local import, and
449,294,336-byte peak working set. These are local measurements;
the platform's upload validator remains the authority for its CPU and environment.

- ZIP SHA256: d05f01a315b7e8e8a8a9496319febda5f6cb86ec83f71754bef788d402456080
- Runtime SHA256: 7b70d2ba4e896ec9f699e0d098aaeaa7c2204502c847d273d2b5d8c93153cba8
- Opponent runtime SHA256: d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c

The short screen uses five distinct physical CPU cores and suspends the opponent
between turns. Only >=70% score and >=60% actual wins clears its preliminary gate;
that is not the90% objective or permission to promote a weak result. No long
tournament is justified by correctness or speed evidence alone.
