# Refinement checkpoint, 11 September 2026

This is an unfinished experiment, pushed at the user's explicit request to preserve all work.
It is NOT a promoted release. The upload artifact and private main remain the validated
7eda0fa release (runtime d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c).

The new objective is at least 90% actual wins against that fixed released neural engine,
with draws and losses reported separately. It has not been achieved. More targeted training
or data is authorized if supported by the observed weaknesses.

## Current fused move/hash/check candidate

- Same 75% neural blend and released weights, book, and tablebases as draw-fix-fast.
- Fused move application, incremental Zobrist update from immutable bitboard differences,
  and check detection reduce repeated Python/Numba calls.
- 52,009 legal child transitions match prior state, full hash, and check status; all special
  moves included. 2,400 chained hashes match the independent reference.
- 19 history-bearing depth-6 searches match moves, scores, nodes, ranked scores, TT, history,
  and killers. Paired time 22.7601s before, 21.2352s after (1.0718x). This is not Elo evidence.
- Integration: 30 exact depth-6 searches at blends 0/75/100, fallback and book checks pass.
- make gate passes: ruff, mypy (47 files), two real random-opponent smoke games.
- Draw regression passes 3,468 referee comparisons and four tablebase conversions.
- Full WAC: 234/300 at one second per position.
- Endgame regression FAILS: Lucena ends in threefold repetition. Although fixed-depth
  search matches, changed timing exposes an existing conversion weakness. No promotion.
- No real A/B games or package audit yet for this fused candidate.

## Preserved results

- Released model: +62 =43 -7 in 112 games at 120s+0.5 versus classical 05046b2;
  paired SPRT accepted H1. This does not establish a platform rating.
- draw-fix-fast: +5 =11 -4 in 20 games at 20s+0.3 versus released model, 52.5% score.
- The later long draw-fix test was stopped at the user's request after three completed games;
  it is incomplete and excluded from strength conclusions.
- draw-neural100: +5 =13 -6 in 24 games at 20s+0.3, 47.9167% score. Rejected;
  improved fitting error did not translate to improved match results.
- Earlier slow draw handling and no-ponder attempts also remain unpromoted.

The archive stores code patches against the listed commits, diagnostic sources and results,
and asset hashes for this session's earlier worktrees. Existing published branches preserve
their committed work. Datasets, environments, raw checkpoints, and unselected weights are
excluded as required by the original request. Selected released weights are already in Git.
The original shared checkout's uncommitted work was not touched.

## Next experiments

1. Resolve the uncovered KRPvKR conversion and investigate tablebase/dependency size within
   the 50 MB package cap before changing assets.
2. Improve search efficiency and draw decisions, then repeat correctness/endgame screens.
3. Reweight existing sparse endgames before downloading more broad data; preserve random-init
   lineage, held-out split, and quantization checks for any trained candidate.
4. Use fresh, diverse paired openings for short real-clock screens. Spend time on full-clock
   confirmation only after a substantial measured gain. No claim that 90% wins is guaranteed.
